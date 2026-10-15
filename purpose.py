# -*- coding: utf-8 -*-
"""
purpose.py —— 出行目的识别（课时 43~46）

入口 `purpose_analysis()`：读 stage3_label.csv → 产出 purpose_result.csv。

分析单元不再是「单个点」而是「驻留段」——连续一串停留点代表一次在某地的活动：
  1. 切段：连续 label=0 属于同一段，遇到 label=1（或换用户）开新段。
     与阶段1「位置一变就切段」不同，这里只由出行点打断，因为小范围活动已被阶段3 判为驻留；
     单点驻留段也保留（一次短暂活动同样值得推断目的）；
  2. 段聚合：起止时间、点数、中心经纬度（段内点均值，比任意单点更稳）、时长、起止小时；
  3. POI 最近邻匹配：段中心 × 全部 POI 广播算距离矩阵，取最近 POI；
     最近距离 < PURPOSE_MATCH_THRESHOLD 才算有效空间证据；
  4. decide_purpose：时间定居住 → 空间证据定购物/旅游 → 时间兜底（先窄后宽）。

关于第 4 步的「窗口覆盖陷阱」（课时 45）：旅游窗 [9,17] 是购物窗 [9,21] 的子集，
若按讲义原始优先级先判购物，旅游分支永远不可达（实跑 旅游恒为 0）。
本模块按讲义修正版实现：白天活动优先采信最近 POI 类型，兜底时先判更窄的 9-17。

`python purpose.py` 单独运行本模块（需先跑过阶段3）。
"""
import numpy as np
import pandas as pd

import config
from utils import (drop_out_of_range_poi, haversine, is_night, log, log_step, parse_time,
                   read_csv, read_csv_gbk, save_csv, Timer)

PURPOSE_COLS = ["imsi", "seg_id", "start_time", "end_time", "points",
                "center_lon", "center_lat", "duration_hours", "start_hour", "end_hour",
                "final_purpose", "nearest_poi_name", "nearest_poi_type",
                "nearest_dist_m", "match_success"]


def split_stay_segments(df):
    """课时 43：出行点 label=1 打断驻留段（「变化标记 + groupby cumsum」通用切段模式）。"""
    user_change = (df["imsi"] != df["imsi"].shift(1)).astype(int)
    break_by_travel = (df["label"] == 1).astype(int)
    df = df.copy()
    df["stay_seg"] = (user_change + break_by_travel).groupby(df["imsi"]).cumsum()

    stay = df[df["label"] == 0]
    agg = stay.groupby(["imsi", "stay_seg"], as_index=False).agg(
        start_time=("time_dt", "min"),
        end_time=("time_dt", "max"),
        points=("time", "count"),
        center_lon=("lon", "mean"),
        center_lat=("lat", "mean"),
    )
    agg["duration_hours"] = (agg["end_time"] - agg["start_time"]).dt.total_seconds() / 3600.0
    agg["start_hour"] = agg["start_time"].dt.hour
    agg["end_hour"] = agg["end_time"].dt.hour
    log(f"共切出 {len(agg)} 个驻留段（含单点段，不过滤），覆盖 {agg['imsi'].nunique()} 个用户")
    return agg


def load_poi():
    """读 出行目的静态数据.csv（GBK，真实列序 lon,lat,poi_type,poi_name）。"""
    poi = read_csv_gbk(config.PURPOSE_POI_FILE, header=None,
                       names=["lon", "lat", "poi_type", "poi_name"])
    for c in ("lon", "lat"):
        poi[c] = pd.to_numeric(poi[c], errors="coerce")
    poi = poi.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    poi = drop_out_of_range_poi(poi, "目的POI表")
    log(f"POI 静态表 {len(poi):,} 条，类型分布 {poi['poi_type'].value_counts().to_dict()}")
    return poi


def match_nearest_poi(agg, poi):
    """课时 46：段中心 × 全部 POI 广播算距离矩阵，分批控制内存峰值。"""
    poi_lon = poi["lon"].to_numpy()
    poi_lat = poi["lat"].to_numpy()
    n = len(agg)
    nearest_idx = np.empty(n, dtype=int)
    nearest_dist = np.empty(n)
    for s in range(0, n, config.BATCH_SIZE):        # 分批广播：批大小 × POI 数
        e = min(s + config.BATCH_SIZE, n)
        c_lon = agg["center_lon"].to_numpy()[s:e, None]      # (b,1)
        c_lat = agg["center_lat"].to_numpy()[s:e, None]
        dist_m = haversine(c_lon, c_lat, poi_lon[None, :], poi_lat[None, :], unit="m")
        idx = dist_m.argmin(axis=1)
        nearest_idx[s:e] = idx
        nearest_dist[s:e] = dist_m[np.arange(e - s), idx]

    agg = agg.copy()
    agg["nearest_poi_name"] = poi.loc[nearest_idx, "poi_name"].to_numpy()
    agg["nearest_poi_type"] = poi.loc[nearest_idx, "poi_type"].to_numpy()
    agg["nearest_dist_m"] = np.round(nearest_dist, 1)
    agg["match_success"] = nearest_dist < config.PURPOSE_MATCH_THRESHOLD
    log(f"POI空间匹配成功: {int(agg['match_success'].sum())}/{n}"
        f"（阈值{config.PURPOSE_MATCH_THRESHOLD}m）")
    return agg


def decide_purpose(row):
    """课时 45 修正版：时间定居住 → POI 定购物/旅游 → 时间兜底（先窄后宽，避免覆盖）。"""
    s, e, dur = row["start_hour"], row["end_hour"], row["duration_hours"]
    # 1) 居住：长时间停留且端点落在夜间（在家过夜的典型特征）
    if dur > config.LIVING_DURATION_THRESHOLD and (is_night(s) or is_night(e)):
        return "居住"
    # 2) 空间证据：1000m 内最近 POI 是购物/旅游，直接采用（时间本身区分不了这两类）
    poi_type = row.get("nearest_poi_type")
    if bool(row.get("match_success", False)) and poi_type in ("购物", "旅游"):
        return poi_type
    # 3) 时间兜底：先判更窄的 9-17，再判 9-21
    if s >= config.TRAVEL_WINDOW[0] and e <= config.TRAVEL_WINDOW[1]:
        return "旅游"
    if s >= config.SHOP_WINDOW[0] and e <= config.SHOP_WINDOW[1]:
        return "购物"
    return "其他"


def purpose_analysis(df=None):
    """执行出行目的识别，保存 purpose_result.csv 并返回驻留段结果表。"""
    log_step(5, "出行目的识别（驻留段 + POI 最近邻 + 时间规则）")
    if df is None:
        df = read_csv(config.STAGE3_LABEL_FILE,
                      dtype={"imsi": str, "lac_id": str, "cell_id": str})
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.sort_values(["imsi", "time"]).reset_index(drop=True).copy()
    df["time_dt"] = parse_time(df["time"])

    agg = split_stay_segments(df)
    agg = match_nearest_poi(agg, load_poi())
    agg["final_purpose"] = agg.apply(decide_purpose, axis=1)
    log("出行目的分布: " + ", ".join(f"{k}={v}" for k, v in
        agg["final_purpose"].value_counts().items()))

    out = agg.rename(columns={"stay_seg": "seg_id"}).copy()
    out["start_time"] = out["start_time"].dt.strftime("%Y%m%d%H%M%S")
    out["end_time"] = out["end_time"].dt.strftime("%Y%m%d%H%M%S")
    out["center_lon"] = out["center_lon"].round(6)
    out["center_lat"] = out["center_lat"].round(6)
    out["duration_hours"] = out["duration_hours"].round(2)
    save_csv(out[PURPOSE_COLS], config.PURPOSE_RESULT_FILE)
    return out


if __name__ == "__main__":
    with Timer("出行目的识别"):
        result = purpose_analysis()
    log(f"结果预览：\n{result.head(3).to_string(index=False)}")
