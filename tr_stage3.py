# -*- coding: utf-8 -*-
"""
tr_stage3.py —— 出行/驻留识别·阶段3：规则精判（课时 39~42）

入口 `stage3_analysis()`：读 cleaned_data.csv + stage2_cluster.csv → 产出 stage3_label.csv。

阶段1 求召回、阶段2 看结构，到这里用「空间展布 + 时间跨度」把误判纠正回来：
  1. 双键 [imsi, time] 左连接，把簇标签贴回清洗全量点（不同用户同一时刻都有点，不能只用 time）；
  2. 未参与聚类的点 cluster = -1，label 默认 0（驻留）——保守策略，宁可少判出行，
     不把停留误判成赶路；
  3. 单点簇（阶段2 已标 is_single_cluster=1）直接判出行：孤立高速点没有任何停留证据；
  4. 多点簇：簇中心 = 点坐标均值，取首/中/尾三点算到中心的 Haversine 距离（米），
     再算首末时间跨度（小时）。三距都 < CLUSTER_DIST_THRESHOLD 且跨度 < TIME_SPAN_THRESHOLD
     → 驻留，否则整簇出行。

聚类用经纬度平面欧氏距离，判阈值用真实 Haversine 米值——「聚类用近似、判阈值用真实」的分工。

`python tr_stage3.py` 单独运行本模块（需先跑过阶段1、2）。
"""
import numpy as np
import pandas as pd

import config
from utils import haversine, log, log_step, parse_time, read_csv, save_csv, Timer


def attach_clusters(cleaned, cluster):
    """课时 40：把簇标签左连接回全量点，未聚类的点 cluster=-1。"""
    df = cleaned.merge(cluster[["imsi", "time", "cluster", "is_single_cluster"]],
                       on=["imsi", "time"], how="left")
    assert len(df) == len(cleaned), "左连接后行数必须不变，否则右表的 (imsi,time) 有重复"
    df["cluster"] = df["cluster"].fillna(-1).astype(int)
    df["is_single_cluster"] = df["is_single_cluster"].fillna(0).astype(int)
    df["time_dt"] = parse_time(df["time"])
    df = df.sort_values(["imsi", "time"]).reset_index(drop=True)
    df["label"] = 0                                    # 默认全部驻留
    log(f"簇标签合并回全量：{len(df):,} 点，其中参与聚类 "
        f"{int((df['cluster'] >= 0).sum()):,} 点、未参与（默认驻留） "
        f"{int((df['cluster'] < 0).sum()):,} 点")
    return df


def judge(df):
    """课时 40~41：逐簇判定 label。返回 (df, 统计字典, 多点簇明细表)。"""
    clustered = df[df["cluster"] >= 0]
    n_single_travel = n_multi_travel = n_multi_stay = 0
    detail = []

    for (imsi, cid), g in clustered.groupby(["imsi", "cluster"], sort=False):
        idx = g.index
        if int(g["is_single_cluster"].iloc[0]) == 1 or len(g) == 1:
            df.loc[idx, "label"] = 1                   # 单点簇：孤立高速点，直接判出行
            n_single_travel += 1
            continue

        center_lon = g["lon"].mean()
        center_lat = g["lat"].mean()
        first, mid, last = g.iloc[0], g.iloc[len(g) // 2], g.iloc[-1]

        d_first = haversine(center_lon, center_lat, first["lon"], first["lat"], unit="m")
        d_mid = haversine(center_lon, center_lat, mid["lon"], mid["lat"], unit="m")
        d_last = haversine(center_lon, center_lat, last["lon"], last["lat"], unit="m")
        time_span_h = (last["time_dt"] - first["time_dt"]).total_seconds() / 3600.0

        is_stay = (d_first < config.CLUSTER_DIST_THRESHOLD
                   and d_mid < config.CLUSTER_DIST_THRESHOLD
                   and d_last < config.CLUSTER_DIST_THRESHOLD
                   and time_span_h < config.TIME_SPAN_THRESHOLD)
        if is_stay:
            n_multi_stay += 1                          # 保持 label=0
        else:
            df.loc[idx, "label"] = 1
            n_multi_travel += 1
        detail.append({"imsi": imsi, "cluster": cid, "points": len(g),
                       "d_first_m": round(float(d_first), 1),
                       "d_mid_m": round(float(d_mid), 1),
                       "d_last_m": round(float(d_last), 1),
                       "time_span_h": round(time_span_h, 2),
                       "判定": "驻留" if is_stay else "出行"})

    stats = {"单点簇判出行": n_single_travel,
             "多点簇判出行": n_multi_travel,
             "多点簇判驻留": n_multi_stay}
    return df, stats, pd.DataFrame(detail)


def stage3_analysis(cleaned=None, cluster=None):
    """执行阶段3，保存 stage3_label.csv 并返回全量点的 0/1 标签表。"""
    log_step(4, "阶段3：三点距离 + 时间跨度规则精判")
    if cleaned is None:
        cleaned = read_csv(config.CLEANED_DATA_FILE,
                           dtype={"imsi": str, "lac_id": str, "cell_id": str})
    if cluster is None:
        cluster = read_csv(config.STAGE2_CLUSTER_FILE,
                           dtype={"imsi": str, "lac_id": str, "cell_id": str})
        for c in ("lon", "lat", "cluster", "is_single_cluster"):
            cluster[c] = pd.to_numeric(cluster[c], errors="coerce")

    df = attach_clusters(cleaned.copy(), cluster)
    df, stats, detail = judge(df)

    n_travel = int((df["label"] == 1).sum())
    n_stay = int((df["label"] == 0).sum())
    log(f"单点簇判出行 {stats['单点簇判出行']}；多点簇判出行 {stats['多点簇判出行']}；"
        f"多点簇判驻留 {stats['多点簇判驻留']}")
    log(f"最终：出行 {n_travel:,}（{n_travel / len(df) * 100:.1f}%），"
        f"驻留 {n_stay:,}（{n_stay / len(df) * 100:.1f}%）")
    assert set(df["label"].unique()) <= {0, 1}

    out = df[["imsi", "time", "lac_id", "cell_id", "lon", "lat", "label"]].copy()
    save_csv(out, config.STAGE3_LABEL_FILE)
    return out


if __name__ == "__main__":
    with Timer("阶段3规则精判"):
        result = stage3_analysis()
    log(f"标签分布：{result['label'].value_counts().sort_index().to_dict()}")
