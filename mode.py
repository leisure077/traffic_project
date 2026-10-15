# -*- coding: utf-8 -*-
"""
mode.py —— 出行方式识别（课时 47~50）

入口 `mode_analysis()`：读 stage3_label.csv → 产出 mode_result.csv。

分析单元是「出行段」——连续一串 label=1 的点，遇到驻留点（或换用户）开新段：
  1. 过滤单点出行段：只有 1 个点既算不出段内速度，也谈不上一次完整出行；
  2. 段内重算相邻点速度取平均（课时 48）：阶段1 的 speed 是全表相邻速度、混入了驻留点，
     段内均速更贴近这次出行的实际速度。时间差用 np.diff 而不是 Series.diff()，
     长度才是 n-1、与相邻距离严格对齐（早期写法报 shapes (2,) (3,) 就是这个坑）；
  3. 站点空间证据（课时 49）：统计段内有多少个点落在地铁站 / 公交站 500m 内。
     对每个点取「到最近站点的距离」再判一次，所以一个点旁边有 3 个公交站也只记 1；
  4. 综合裁决（课时 50）：地铁 ≥2 点 → 公交 ≥3 点 → 均速 >20 → 其他。
     站点证据排在速度前面，是因为地铁与小汽车的速度区间重叠，只有空间证据能把它们分开。

信令是基站级定位（误差数百米到数公里），方式推断只能给「沿线概率大」的粗结论。

`python mode.py` 单独运行本模块（需先跑过阶段3）。
"""
import numpy as np
import pandas as pd

import config
from utils import (drop_out_of_range_poi, format_time, haversine, log, log_step, parse_time,
                   read_csv, read_csv_gbk, save_csv, Timer)

MODE_COLS = ["imsi", "seg_id", "points", "start_time", "end_time",
             "start_lon", "start_lat", "end_lon", "end_lat",
             "distance_km", "avg_speed", "subway_points", "bus_points", "final_mode"]


def split_travel_segments(df):
    """课时 47：驻留点 label=0 打断出行段。"""
    user_change = (df["imsi"] != df["imsi"].shift(1)).astype(int)
    break_by_stay = (df["label"] == 0).astype(int)
    df = df.copy()
    df["trip_seg"] = (user_change + break_by_stay).groupby(df["imsi"]).cumsum()
    travel = df[df["label"] == 1]
    n_seg = travel.groupby(["imsi", "trip_seg"]).ngroups
    log(f"{len(travel):,} 个出行点共切成 {n_seg} 个出行段"
        f"（其中单点段 {int((travel.groupby(['imsi', 'trip_seg']).size() == 1).sum())} 个）")
    return travel


def load_stations():
    """读 出行方式静态数据.csv（GBK，列序 lon,lat,mode_type,station,line），拆地铁/公交。

    用 contains 而不是等于，兼容「地铁站」「公交首末站」等写法。
    """
    st = read_csv_gbk(config.MODE_STATION_FILE, header=None,
                      names=["lon", "lat", "mode_type", "station_name", "line_name"])
    for c in ("lon", "lat"):
        st[c] = pd.to_numeric(st[c], errors="coerce")
    st = st.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    st = drop_out_of_range_poi(st, "方式站点表")
    kind = st["mode_type"].astype(str)
    subway = st[kind.str.contains("地铁", na=False)].reset_index(drop=True)
    bus = st[kind.str.contains("公交", na=False)].reset_index(drop=True)
    log(f"地铁站 {len(subway)} 个，公交站 {len(bus)} 个")
    return subway, bus


def count_near_points(point_lon, point_lat, st_lon, st_lat, threshold=None):
    """课时 49：段内有多少个点落在站点 threshold 米内（每个点只计一次）。"""
    threshold = config.MODE_MATCH_THRESHOLD if threshold is None else threshold
    if len(st_lon) == 0 or len(point_lon) == 0:
        return 0
    d = haversine(np.asarray(point_lon)[:, None], np.asarray(point_lat)[:, None],
                  np.asarray(st_lon)[None, :], np.asarray(st_lat)[None, :], unit="m")
    nearest = d.min(axis=1)                 # 每个点到最近站点的距离
    return int((nearest < threshold).sum())


def decide_mode(sub_pts, bus_pts, avg_speed):
    """课时 50：先地铁（证据强）、再公交（站点密、门槛更高）、再小汽车（速度兜底）、最后其他。"""
    if sub_pts >= config.SUBWAY_MIN_POINTS:
        return "地铁"
    if bus_pts >= config.BUS_MIN_POINTS:
        return "公交"
    if avg_speed > config.MODE_BUS_SPEED:
        return "小汽车"
    return "其他"


def mode_analysis(df=None):
    """执行出行方式识别，保存 mode_result.csv 并返回出行段结果表。"""
    log_step(6, "出行方式识别（段内均速 + 地铁/公交站点证据）")
    if df is None:
        df = read_csv(config.STAGE3_LABEL_FILE,
                      dtype={"imsi": str, "lac_id": str, "cell_id": str})
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.sort_values(["imsi", "time"]).reset_index(drop=True).copy()
    df["time_dt"] = parse_time(df["time"])

    travel = split_travel_segments(df)
    subway, bus = load_stations()
    sub_xy = (subway["lon"].to_numpy(), subway["lat"].to_numpy())
    bus_xy = (bus["lon"].to_numpy(), bus["lat"].to_numpy())

    n_drop_single = n_drop_speed = 0
    records = []
    n_seg = travel.groupby(["imsi", "trip_seg"]).ngroups
    log(f"逐段判定（共{n_seg}段）")
    for (imsi, seg), g in travel.groupby(["imsi", "trip_seg"], sort=False):
        if len(g) < 2:
            n_drop_single += 1
            continue
        g = g.sort_values("time_dt")
        lon = g["lon"].to_numpy()
        lat = g["lat"].to_numpy()

        secs = np.diff(g["time_dt"].to_numpy()).astype("timedelta64[s]").astype(float)
        d_km = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:], unit="km")
        with np.errstate(divide="ignore", invalid="ignore"):
            spd = d_km * 3600.0 / secs
        valid = (spd > 0) & (spd < config.ABNORMAL_SPEED)   # 去掉 0 间隔与基站跳变
        if not valid.any():
            n_drop_speed += 1
            continue
        avg_speed = float(np.mean(spd[valid]))

        first, last = g.iloc[0], g.iloc[-1]
        distance_km = float(haversine(first["lon"], first["lat"],
                                      last["lon"], last["lat"], unit="km"))
        sub_pts = count_near_points(lon, lat, sub_xy[0], sub_xy[1])
        bus_pts = count_near_points(lon, lat, bus_xy[0], bus_xy[1])

        records.append({
            "imsi": imsi, "seg_id": seg, "points": len(g),
            "start_time": format_time(first["time_dt"]),
            "end_time": format_time(last["time_dt"]),
            "start_lon": round(float(first["lon"]), 6), "start_lat": round(float(first["lat"]), 6),
            "end_lon": round(float(last["lon"]), 6), "end_lat": round(float(last["lat"]), 6),
            "distance_km": round(distance_km, 3),
            "avg_speed": round(avg_speed, 2),
            "subway_points": sub_pts, "bus_points": bus_pts,
            "final_mode": decide_mode(sub_pts, bus_pts, avg_speed),
        })

    result = pd.DataFrame(records, columns=MODE_COLS)
    log(f"有效出行段 {len(result)}（丢弃单点段 {n_drop_single}、无有效速度段 {n_drop_speed}）")
    log("出行方式分布: " + ", ".join(f"{k}={v}" for k, v in
        result["final_mode"].value_counts().items()))
    save_csv(result, config.MODE_RESULT_FILE)
    return result


if __name__ == "__main__":
    with Timer("出行方式识别"):
        out = mode_analysis()
    log(f"结果预览：\n{out.head(3).to_string(index=False)}")
