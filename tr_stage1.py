# -*- coding: utf-8 -*-
"""
tr_stage1.py —— 出行/驻留识别·阶段1：低频点标记 + 轨迹切段 + 速度粗筛（课时 31~34）

入口 `stage1_analysis()`：读 cleaned_data.csv → 产出 stage1_travel_points.csv。

粗筛只负责「求召回」，宁多勿漏：
  1. 低频点（课时 32）：同一用户在同一坐标出现 < LOW_FREQ_THRESHOLD 次，多半是基站切换
     时的短暂路过或漂移。讲义口径是反连接删除，本项目改成**只标记 is_low 不删除**——
     实测这些一次性坐标上压着 83.7% 的高速移动点，删掉就把出行样本删没了（详见 README 七）；
     阶段3 做最终判定时用清洗全量点，保证标签覆盖全量；
  2. 轨迹切段（课时 32）：换用户或坐标一变就开新段，段内坐标相同，段间那条边就是潜在移动；
  3. 相邻点速度（课时 33）：groupby(imsi) 内取上一条坐标与时间，Haversine 距离 / 时间差；
     速度 <0 或 >ABNORMAL_SPEED 视为基站跳跃，置 NaN 而不是删行——该点的位置和时间仍有效；
  4. 粗判：speed > SPEED_THRESHOLD 的到达点，加上它对应的出发点，一起作为「疑似出行点」
     输出交给阶段2 聚类——一次移动由前后两个位置共同证明，只标终点会把出发地当成驻留。

`python tr_stage1.py` 单独运行本模块（需先跑过 data_clean.py）。
"""
import numpy as np
import pandas as pd

import config
from utils import haversine, log, log_step, parse_time, read_csv, save_csv, Timer


def mark_low_freq(df):
    """课时 32 第 2 步的低频统计照做，但只打 is_low 标记，不再删点。

    删点的实测代价：速度>8km/h 的点里 83.7% 落在「同用户同坐标出现<4 次」的基站上——
    那正是移动途中只驻留一次的基站，整批删掉后出行率从 25.6% 被压到 3.1%。
    """
    loc_cnt = df.groupby(["imsi", "lon", "lat"]).size().reset_index(name="cnt")
    low_freq = loc_cnt[loc_cnt["cnt"] < config.LOW_FREQ_THRESHOLD][["imsi", "lon", "lat"]].copy()
    low_freq["_is_low"] = 1
    df = df.merge(low_freq, on=["imsi", "lon", "lat"], how="left")
    df["is_low"] = df["_is_low"].notna().astype("int64")
    df = df.drop(columns="_is_low")
    n_low = int(df["is_low"].sum())
    log(f"低频点标记（同用户同坐标 <{config.LOW_FREQ_THRESHOLD} 次）："
        f"{len(df):,} 行中 {n_low:,} 个标记 is_low=1，全部保留不删除")
    return df


def split_segments(df):
    """课时 32 第 3 步：换用户 或 坐标变化 → 段号 +1；groupby(imsi) 保证用户之间互不干扰。

    段号在每个用户内部从 1 递增，所以段号本身会跨用户重复，统计段数要用 (imsi, seg_id)。
    """
    user_change = df["imsi"] != df["imsi"].shift(1)
    loc_change = (df["lon"] != df["lon"].shift(1)) | (df["lat"] != df["lat"].shift(1))
    seg_change = (user_change | loc_change).astype("int64")
    df = df.copy()
    df["seg_id"] = seg_change.groupby(df["imsi"]).cumsum()
    seg_size = df.groupby(["imsi", "seg_id"]).size()
    log(f"轨迹切段：{df.groupby(['imsi', 'seg_id']).ngroups} 个轨迹段，"
        f"其中单点段 {int((seg_size == 1).sum())} 个")
    return df


def compute_speed(df):
    """课时 33：同一用户内相邻点的 Haversine 距离 / 时间差 = 速度（km/h）。

    groupby(imsi) 保证不会把 A 用户的尾点和 B 用户的头点算成一条边；
    每个用户第一条 prev 为 NaN，速度自然为 NaN，不会误判。
    """
    df = df.copy()
    df["time_diff"] = df.groupby("imsi")["time_dt"].diff().dt.total_seconds()
    prev_lon = df.groupby("imsi")["lon"].shift(1)
    prev_lat = df.groupby("imsi")["lat"].shift(1)
    dist_km = haversine(prev_lon, prev_lat, df["lon"], df["lat"], unit="km")
    df["dist_km"] = dist_km
    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed"] = dist_km * 3600.0 / df["time_diff"]

    abnormal = (df["speed"] < 0) | (df["speed"] > config.ABNORMAL_SPEED)
    n_abnormal = int(abnormal.sum())
    df.loc[abnormal, "speed"] = np.nan          # 置空而非删行：位置与时间仍然有效
    log(f"速度计算：可算速度点 {int(df['speed'].notna().sum()):,} 个，"
        f"其中 {n_abnormal} 个 <0 或 >{config.ABNORMAL_SPEED} km/h 判为基站跳跃已置 NaN")
    return df


def select_travel_points(df):
    """课时 33/34 粗筛输出：一次移动的**出发点和到达点都算疑似出行**。

    只标到达点等于宣布「出发地那条记录是驻留」，可人当时已经离开这个基站了；
    信令点本身不含运动信息，一次移动是由「前一个位置 + 后一个位置」这一对点共同证明的。
    这里仍只求召回，误标的点交给阶段2/3 判回驻留。
    """
    df = df.copy()
    arrive = (df["speed"] > config.SPEED_THRESHOLD).to_numpy()
    same_user_next = (df["imsi"] == df["imsi"].shift(-1)).fillna(False).to_numpy()
    depart = np.zeros(len(df), dtype=bool)
    depart[:-1] = arrive[1:]                      # 下一点是移动到达点，本点就是它的出发地
    depart &= same_user_next                      # 不跨用户配对
    df["is_travel_stage1"] = arrive | depart
    log(f"速度粗筛（>{config.SPEED_THRESHOLD} km/h）：到达点 {int(arrive.sum()):,} 个"
        f" ＋ 只作出发地的点 {int((depart & ~arrive).sum()):,} 个"
        f" ＝ 疑似出行点 {int(df['is_travel_stage1'].sum()):,} 个，"
        f"占本阶段 {len(df):,} 个点的 {df['is_travel_stage1'].mean() * 100:.1f}%")
    return df


def stage1_analysis(df=None):
    """执行阶段1，保存 stage1_travel_points.csv 并返回疑似出行点表。"""
    log_step(2, "阶段1：低频点标记 + 轨迹切段 + 速度粗筛")
    if df is None:
        df = read_csv(config.CLEANED_DATA_FILE,
                      dtype={"imsi": str, "lac_id": str, "cell_id": str})
        log(f"读入清洗结果 {len(df):,} 行 / {df['imsi'].nunique()} 个用户")
    df = df.sort_values(["imsi", "time"]).reset_index(drop=True).copy()
    df["time_dt"] = parse_time(df["time"])

    df = mark_low_freq(df)
    df = split_segments(df)
    df = compute_speed(df)
    df = select_travel_points(df)

    travel = df[df["is_travel_stage1"]].copy()
    if len(travel):
        fast = travel[travel["speed"].notna()]
        log("出行点速度分布：min %.2f ｜ 中位 %.2f ｜ 均值 %.2f ｜ max %.2f km/h"
            % (fast["speed"].min(), fast["speed"].median(),
               fast["speed"].mean(), fast["speed"].max()))

    out_cols = ["imsi", "time", "lac_id", "cell_id", "lon", "lat", "speed"]
    out = travel[out_cols].copy()
    out["speed"] = out["speed"].round(2)
    save_csv(out, config.STAGE1_TRAVEL_POINTS_FILE)
    return out


if __name__ == "__main__":
    with Timer("阶段1速度粗筛"):
        result = stage1_analysis()
    log(f"结果预览：\n{result.head(3).to_string(index=False)}")
