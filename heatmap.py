# -*- coding: utf-8 -*-
"""
heatmap.py —— 商圈实时热力：10 分钟滑动窗口人流量（课时 51~54）

入口 `heatmap_analysis()`：读 实时流数据.csv + 商业区数据.csv → 产出 heatmap_result.csv
（10 分钟窗口长表）和 heat_hourly.csv（商圈 × 24 小时真实人次，图5 用它）。

  1. 读实时流：真实文件是 4 列、已自带经纬度（原讲义说只有 3 列要关联基站，以真实数据为准），
     过滤目标日、解析时间；
  2. 读商圈：真实列序 lon,lat,biz_name；
  3. 分批广播匹配最近商圈：25.6 万点 × 104 商圈的距离矩阵一次算要占 2668 万个数，
     按 BATCH_SIZE 分批把内存压到「批大小 × 商圈数」；命中 < HEATMAP_MATCH_THRESHOLD 才算在该商圈；
  4. 分钟 × 商圈计数 → reindex 补齐全天 1440 分钟 → rolling(HEATMAP_WINDOW_SIZE) 滑动求和。
     不补 0，中间断档的分钟会被 rolling 当成连续，窗口结果就错了；
  5. 宽转长输出 [time, biz_name, count]，丢掉 count=0 的行减小文件。

「最近 10 分钟人数」用 sum 而不是 mean：要的是窗口内的累计人次，不是平均密度。

`python heatmap.py` 单独运行本模块（只依赖静态数据，不需要先跑前面几个模块）。
"""
import numpy as np
import pandas as pd

import config
from utils import (drop_out_of_range_poi, haversine, log, log_step, read_csv, save_csv,
                   Timer)


def load_stream():
    """课时 51：读实时流（4 列，已含经纬度），过滤目标日。"""
    stream = read_csv(config.STREAM_DATA_FILE, header=None,
                      names=["time", "imsi", "lon", "lat"], dtype={"time": str, "imsi": str})
    raw_rows = len(stream)
    for c in ("lon", "lat"):
        stream[c] = pd.to_numeric(stream[c], errors="coerce")
    stream = stream.dropna(subset=["lon", "lat"])
    stream = stream[stream["time"].astype(str).str.startswith(config.TARGET_DATE)]
    stream = stream.copy()
    stream["time_dt"] = pd.to_datetime(stream["time"], format="%Y%m%d%H%M%S", errors="coerce")
    stream = stream.dropna(subset=["time_dt"]).reset_index(drop=True)
    log(f"实时流读入 {raw_rows:,} 行 → 当日有效点 {len(stream):,}，时间范围 "
        f"{stream['time_dt'].min()} ~ {stream['time_dt'].max()}")
    return stream


def load_business():
    """课时 51：读商圈（真实列序 lon,lat,biz_name），剔掉范围外的脏坐标。"""
    biz = read_csv(config.BIZ_DATA_FILE, header=None, names=["lon", "lat", "biz_name"])
    for c in ("lon", "lat"):
        biz[c] = pd.to_numeric(biz[c], errors="coerce")
    biz = biz.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    biz = drop_out_of_range_poi(biz, "商圈表")
    log(f"商圈 POI {len(biz)} 个，商圈名 {sorted(biz['biz_name'].unique())}")
    return biz


def match_business(stream, biz):
    """课时 52：分批广播算每个流点的最近商圈，只保留命中阈值内的点。"""
    s_lon = stream["lon"].to_numpy()
    s_lat = stream["lat"].to_numpy()
    b_lon = biz["lon"].to_numpy()
    b_lat = biz["lat"].to_numpy()
    n = len(stream)

    nearest_idx = np.full(n, -1, dtype=int)
    for s in range(0, n, config.BATCH_SIZE):
        e = min(s + config.BATCH_SIZE, n)
        d = haversine(s_lon[s:e, None], s_lat[s:e, None], b_lon[None, :], b_lat[None, :],
                      unit="m")
        idx = d.argmin(axis=1)
        dist = d[np.arange(e - s), idx]
        nearest_idx[s:e] = np.where(dist < config.HEATMAP_MATCH_THRESHOLD, idx, -1)

    keep = nearest_idx >= 0
    stream = stream.loc[keep].copy()
    stream["biz_name"] = biz["biz_name"].to_numpy()[nearest_idx[keep]]
    log(f"命中 {int(keep.sum()):,} 点（阈值 {config.HEATMAP_MATCH_THRESHOLD}m），"
        f"覆盖 {stream['biz_name'].nunique()} 个商圈")
    # 零命中的商圈会被 groupby 静默丢掉，结果里看不出少过东西，所以这里点名说清楚
    hit = set(stream["biz_name"].unique())
    zero = [str(b) for b in biz["biz_name"].unique() if b not in hit]
    if zero:
        log(f"以下商圈 {config.HEATMAP_MATCH_THRESHOLD}m 内全天无流点，不出现在结果表中："
            + "、".join(zero), level="WARN")
    return stream


def hourly_counts(stream):
    """按 (商圈, 小时) 统计真实命中人次——图5 画的就是这张表。

    必须在 rolling_window 之前算：heatmap_result.csv 里一小时有 60 个互相重叠的 10 分钟
    窗口，按小时加起来等于每个人被数 10 次。匹配口径与长表完全一致（最近商圈、阈值内才算命中）。
    行不做类别筛选：Day10 规定图5 是「Top10 商圈 × 24 小时」，商圈表一共 5 个商圈，全给。
    """
    s = stream.copy()
    s["hour"] = s["time_dt"].dt.hour
    pivot = s.groupby(["biz_name", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    pivot["全天合计"] = pivot.sum(axis=1)
    # 商圈名还原成普通列：utils.save_csv 写 index=False，留着索引会只存数字，认不出谁是谁
    pivot = pivot.sort_values("全天合计", ascending=False).reset_index()
    log(f"小时命中表：{pivot.shape[0]} 个商圈 × 24 小时，"
        f"全天合计 {int(pivot['全天合计'].sum()):,} 人次")
    return pivot


def rolling_window(stream):
    """课时 53：分钟×商圈计数 → 补齐 1440 分钟 → 10 分钟滑动求和 → 转长表。"""
    stream = stream.copy()
    stream["minute"] = stream["time_dt"].dt.floor("1min")
    panel = stream.groupby(["minute", "biz_name"]).size().unstack(fill_value=0)

    day = pd.Timestamp(f"{config.TARGET_DATE[:4]}-{config.TARGET_DATE[4:6]}-{config.TARGET_DATE[6:]}")
    full_minutes = pd.date_range(day, periods=24 * 60, freq="1min")
    panel = panel.reindex(full_minutes, fill_value=0)

    roll = panel.rolling(config.HEATMAP_WINDOW_SIZE, min_periods=1).sum()
    result = roll.stack().reset_index()
    result.columns = ["time", "biz_name", "count"]
    result["count"] = result["count"].astype(int)
    result["time"] = result["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result = result[result["count"] > 0].reset_index(drop=True)
    log(f"结果 {len(result):,} 条（{config.HEATMAP_WINDOW_SIZE} 分钟窗口）")
    return result


def heatmap_analysis():
    """执行实时热力分析，保存 heatmap_result.csv（长表）+ heat_hourly.csv（图5 用），返回长表。"""
    log_step(7, f"实时热力分析（{config.HEATMAP_WINDOW_SIZE} 分钟滑动窗口 × 商圈）")
    stream = match_business(load_stream(), load_business())
    save_csv(hourly_counts(stream), config.HEAT_HOURLY_FILE)   # 趁窗口还没滚，先把人次数出来
    result = rolling_window(stream)

    peak = result.groupby("biz_name")["count"].max().sort_values(ascending=False)
    log(f"{config.HEATMAP_WINDOW_SIZE}分钟窗口人流峰值 Top："
        + ", ".join(f"{name}={int(v)}" for name, v in peak.head(5).items()))
    save_csv(result, config.HEATMAP_RESULT_FILE)
    return result


if __name__ == "__main__":
    with Timer("实时热力分析"):
        out = heatmap_analysis()
    log(f"结果预览：\n{out.head(3).to_string(index=False)}")
