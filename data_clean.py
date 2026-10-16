# -*- coding: utf-8 -*-
"""
data_clean.py —— 流水线第 1 步：数据清洗（课时 23~29，9 条规则）

入口 `clean_data()`：读原始信令 → 9 条规则 → 产出 cleaned_data.csv（六列）。

  规则1 字段抽取      10 列无表头，只取前 4 列（time 列此刻还是毫秒时间戳）
  规则2 去脏 imsi     混入 # * ^ 空格 等字符的记录剔除
  规则3 去空基站编号  lac_id / cell_id 为 NaN 或空串/全空格的剔除
  规则4 时间戳转换    13 位毫秒 UTC 时间戳 → 北京时间 datetime + YYYYMMDDHHMMSS 字符串
  规则5 目标日过滤    只留 config.TARGET_DATE（漏 +8 会把凌晨的点当成前一天的删掉）
 （教学抽样）         课时 28：按整用户抽样约 SAMPLE_ROWS 行，不是随机抽行
  规则6 关联基站      laci = lac_id-cell_id，左连接基站表补 lon/lat，匹配失败删除，
                     再按沈阳经纬度区间（122.5~124.5 / 41.0~42.5）剔除漂移坐标
  规则7 排序          按 imsi、time 升序，保证后续 shift/diff 取到的是同一用户的相邻点
  规则8 分钟去重      同一用户同一分钟的多条上报只留第一条（基站乒乓切换去噪）
  规则9 稀疏用户过滤  当天点数 < MIN_POINTS_PER_USER 的用户整体剔除

`python data_clean.py` 单独运行本模块。
"""
import numpy as np
import pandas as pd

import config
from utils import log, log_step, save_csv, Timer


def read_raw():
    """规则1：无表头读入，全部按字符串读，防止 imsi/lac_id 这类长数字被转成 float 丢精度。"""
    df = pd.read_csv(config.RAW_DATA_FILE, header=None,
                     usecols=list(range(len(config.RAW_COLUMNS))),
                     names=config.RAW_COLUMNS, dtype=str)
    log(f"规则1 字段抽取：读入 {len(df):,} 行（原始 10 列只取前 4 列）")
    return df


def rule2_dirty_imsi(df):
    """规则2：imsi 里混入 # * ^ 空格等字符的记录剔除（课时 24：正则字符集 + 取反）。"""
    before = len(df)
    df = df[~df["imsi"].str.contains(r"[#*^\s]", na=False, regex=True)]
    log(f"规则2 去脏imsi：{before:,} -> {len(df):,}（删除 {before - len(df)}）")
    return df


def rule3_empty_lac_cell(df):
    """规则3：lac_id / cell_id 缺失 NaN 或空串/全空格剔除（两种「空」都要处理）。"""
    before = len(df)
    df = df.dropna(subset=["lac_id", "cell_id"])
    df = df[(df["lac_id"].str.strip() != "") & (df["cell_id"].str.strip() != "")]
    log(f"规则3 去空基站编号：{before:,} -> {len(df):,}（删除 {before - len(df)}）")
    return df


def rule4_timestamp(df):
    """规则4：毫秒 UTC 时间戳 → 北京时间。

    原始 time 列是 13 位毫秒戳（UTC）。直接按 naive 解析等于把北京时间前移 8 小时，
    规则5 就会丢掉当天 0~8 点、混进次日凌晨，时段统计全部失真（课时 25）。
    """
    before = len(df)
    raw = df["time"].astype(str).str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    # 课时 29 的数量级判据要加下界也要加上界：14 位的 YYYYMMDDHHMMSS 日期串同样 >=1e12，
    # 只写下界会把日期串当成毫秒戳去解 epoch，得出 2609 年这类荒谬时间。真实数据是 13 位毫秒戳。
    is_ms = (num.abs() >= 10 ** 12) & (num.abs() < 10 ** 13)

    dt_ms = pd.to_datetime(raw[is_ms].astype("int64"), unit="ms", utc=True).dt.tz_localize(None) \
        + pd.Timedelta(hours=config.RAW_TIMEZONE_OFFSET_HOURS)
    dt_txt = pd.to_datetime(raw[~is_ms], format="%Y%m%d%H%M%S", errors="coerce")
    df["time_dt"] = pd.concat([dt_ms, dt_txt]).sort_index()

    df = df.dropna(subset=["time_dt"])
    df["time"] = df["time_dt"].dt.strftime("%Y%m%d%H%M%S")
    log(f"规则4 毫秒UTC时间戳 +{config.RAW_TIMEZONE_OFFSET_HOURS}时区 转换完成，"
        f"示例: {df['time'].iloc[0]}（{before - len(df)} 行时间不可解析被删除）")
    return df


def rule5_target_date(df):
    """规则5：只保留目标日（time 已是 YYYYMMDDHHMMSS，前 8 位即日期）。"""
    before = len(df)
    df = df[df["time"].str.startswith(config.TARGET_DATE)]
    log(f"规则5 保留 {config.TARGET_DATE}：{before:,} -> {len(df):,}"
        f"（删除跨天 {before - len(df)}）")
    return df


def sample_by_user(df):
    """教学抽样（课时 28）：抽样单位是「用户」而不是「行」。

    随机抽行会把同一用户的连续轨迹抽碎——一个原本 80 点的用户抽完只剩 3 点，
    规则9 之后只剩二十几个超级活跃用户。这里在合格用户里打乱顺序，整用户纳入直到
    累计点数达标，被选中用户的点一颗不拆。

    抽样放在 9 条规则全部跑完之后：原先排在规则6 之前，抽的是"还没补到经纬度、
    还没去重"的行，20000 行过完后面几条只剩 17362 行（产出率 87%），
    少掉的 2638 行是白抽的。全量清洗在本机只多花约 1.6 秒，换来满额样本。
    """
    target = config.SAMPLE_ROWS
    if not target:
        log(f"抽样：SAMPLE_ROWS=0，使用全量 {len(df):,} 行")
        return df
    user_cnt = df.groupby("imsi").size()
    eligible = user_cnt[user_cnt >= config.MIN_POINTS_PER_USER]
    if int(eligible.sum()) <= target:
        log(f"抽样：合格用户总点数 {int(eligible.sum()):,} 未超过目标 {target:,}，不抽样")
        return df
    shuffled = eligible.sample(frac=1.0, random_state=config.RANDOM_STATE)
    kept, total = [], 0
    for imsi, cnt in shuffled.items():
        if total + cnt > target and kept:
            continue                      # 当前用户装不下就跳过，继续找更轻的
        kept.append(imsi)
        total += cnt
        if total >= target:
            break
    before = len(df)
    df = df[df["imsi"].isin(kept)]
    log(f"教学抽样 按整用户：{before:,} -> {len(df):,} 行 / {df['imsi'].nunique()} 个用户"
        f"（目标 {target:,} 行）")
    return df


def rule6_join_station(df):
    """规则6：拼 laci 左连接基站表补经纬度（真实列序 lon,lat,laci），再过滤沈阳范围外的异常坐标。

    坐标过滤出自《项目指导会议纪要》day1 的同一条规则（"基站经纬度补全与坐标过滤"）：
    实测目标日有 1,333 行（0.26%）落在 经度122.5~124.5 / 纬度41.0~42.5 之外，最远的
    lat 到 43.15——那是定位漂移出来的异地基站，留着会把该用户阶段2 的簇心拖偏上百公里。
    """
    before = len(df)
    station = pd.read_csv(config.STATION_DATA_FILE, header=None,
                          names=["lon", "lat", "laci"], dtype=str)
    for c in ("lon", "lat"):
        station[c] = pd.to_numeric(station[c], errors="coerce")
    station = station.dropna(subset=["lon", "lat"]).drop_duplicates("laci", keep="first")

    df = df.copy()
    df["laci"] = df["lac_id"].astype(str) + "-" + df["cell_id"].astype(str)
    df = df.merge(station[["laci", "lon", "lat"]], on="laci", how="left")
    unmatched = int(df["lon"].isna().sum())
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    df["lon"] = df["lon"].astype(float)
    df["lat"] = df["lat"].astype(float)

    in_box = (df["lon"].between(*config.SIGNAL_LON_RANGE)
              & df["lat"].between(*config.SIGNAL_LAT_RANGE))
    outside = int((~in_box).sum())
    users_hit = df.loc[~in_box, "imsi"].nunique()
    df = df[in_box].reset_index(drop=True)
    log(f"规则6 基站关联：{before:,} -> {len(df):,}（关联失败删除 {unmatched} 行，"
        f"坐标越界删除 {outside} 行 / 涉及 {users_hit} 用户，"
        f"基站表去重后 {len(station):,} 个 laci）")
    return df


def rule7_sort(df):
    """规则7：按 imsi、time 升序，段内 shift/diff 才不会串到别的用户或未来时刻。"""
    df = df.sort_values(["imsi", "time"]).reset_index(drop=True)
    log(f"规则7 排序：按 imsi+time 升序，{len(df):,} 行")
    return df


def rule8_dedup_minute(df):
    """规则8：同一用户同一分钟内的多条上报只留第一条（前 12 位 = 到分钟）。"""
    before = len(df)
    df = df.copy()
    df["_minute"] = df["time"].str[:12]
    df = df.drop_duplicates(subset=["imsi", "_minute"], keep="first")
    df = df.drop(columns="_minute").reset_index(drop=True)
    log(f"规则8 分钟去重：{before:,} -> {len(df):,}（删除 {before - len(df)}）")
    return df


def rule9_filter_sparse(df):
    """规则9：当天点数 < MIN_POINTS_PER_USER 的用户整体剔除（样本不足，聚类无意义）。"""
    before = len(df)
    cnt = df.groupby("imsi")["imsi"].transform("size")
    df = df[cnt >= config.MIN_POINTS_PER_USER].reset_index(drop=True)
    log(f"规则9 每人>={config.MIN_POINTS_PER_USER}点：{before:,} -> {len(df):,} 行, "
        f"{df['imsi'].nunique()} 个用户（删除 {before - len(df)}）")
    return df


def clean_data():
    """按 9 条规则清洗原始信令，保存 cleaned_data.csv 并返回 DataFrame。

    每条规则单独计时（照搬老师写法）：全量 53 万行约 3.1 秒，其中规则4 时间戳
    转换占约一半（1.5 秒），删行最多的是规则8 分钟去重（6.7 万行）。
    """
    log_step(1, "数据清洗（9条规则）")
    with Timer("规则1 字段抽取"):
        df = read_raw()
    with Timer("规则2 去脏imsi"):
        df = rule2_dirty_imsi(df)
    with Timer("规则3 去空基站编号"):
        df = rule3_empty_lac_cell(df)
    with Timer("规则4 毫秒时间戳转北京时间"):
        df = rule4_timestamp(df)
    with Timer("规则5 过滤目标日期"):
        df = rule5_target_date(df)
    with Timer("规则6 关联基站补经纬度"):
        df = rule6_join_station(df)
    with Timer("规则7 排序"):
        df = rule7_sort(df)
    with Timer("规则8 分钟去重"):
        df = rule8_dedup_minute(df)
    with Timer("规则9 过滤稀疏用户"):
        df = rule9_filter_sparse(df)
    with Timer("教学抽样"):
        df = sample_by_user(df)            # 非规则，放在全部规则之后

    out = df[config.CLEANED_COLS].copy()
    save_csv(out, config.CLEANED_DATA_FILE)
    log(f"清洗完成：{len(out):,} 点 / {out['imsi'].nunique()} 用户 / "
        f"{out['lon'].round(4).nunique()} 个坐标 / 时间范围 "
        f"{out['time'].min()} ~ {out['time'].max()}")
    return out


if __name__ == "__main__":
    with Timer("数据清洗"):
        result = clean_data()
        log(f"结果预览：\n{result.head(3).to_string(index=False)}")
