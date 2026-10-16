# -*- coding: utf-8 -*-
"""
tr_stage2.py —— 出行/驻留识别·阶段2：手写 K-Means 空间聚类（课时 35~38）

入口 `stage2_analysis()`：读 stage1_travel_points.csv → 产出 stage2_cluster.csv。

全程不调 sklearn（课时 36 要求手写「初始化 → 分配 → 更新 → 收敛」四步）：
  · kmeans(X, k)          随机取 k 个点当初始组长；广播算 (n,k) 距离取 argmin 分配；
                          簇内均值更新中心，空簇保留旧中心避免 NaN；中心位移 < tol 收敛。
  · silhouette_score(D,l) a=簇内均距、b=最近异簇均距，s=(b-a)/max(a,b)，越接近 1 越好；
  · dunn_index(D,l)       簇间最小距离 / 簇内最大直径，越大越好；
  · choose_best_k(X)      K 从 K_MIN 试到 K_MAX，score = 轮廓系数 + DUNN_WEIGHT × 邓恩指数，
                          取最高分，返回 (labels, best_k, sse)。距离矩阵 D 只算一次、各 K 复用。

聚类只在同一个用户的点之间进行（课时 38）——不同用户的点不该聚到一起。
逐用户聚类前 np.random.seed(RANDOM_STATE) 一次，保证结果可复现。

喂给 kmeans 的坐标先经 utils.project_to_meters 换算成米：欧氏距离直接吃经纬度
等于把地图横向压扁约 25%（沈阳纬度下 1° 经度≈83km、1° 纬度≈111km）。

`python tr_stage2.py` 单独运行本模块（需先跑过 tr_stage1.py）。
"""
import numpy as np
import pandas as pd

import config
from utils import log, log_step, pairwise_euclidean, project_to_meters, read_csv, save_csv, Timer


def kmeans(X, k, max_iter=None, tol=None):
    """手写 K-Means（课时 36）。X 形状 (n,2) 的米制平面坐标，返回 (labels, centers)。"""
    max_iter = config.KM_MAX_ITER if max_iter is None else max_iter
    tol = config.KM_TOL if tol is None else tol
    n = len(X)
    if k >= n:                                  # 簇数不少于点数：按「位置」而不是按「点」分簇
        # 同一个基站的点经纬度完全相同，投影后是同一个坐标。按点编号分簇会把两个
        # 一模一样的坐标拆进两簇，所以先去重：同一位置只能有一簇。
        uniq, inv = np.unique(X, axis=0, return_inverse=True)
        return inv.astype(int).reshape(n), uniq

    centers = X[np.random.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))  # (n,k)
        new_labels = dist.argmin(axis=1)

        new_centers = centers.copy()            # 更新步：空簇沿用旧中心，避免均值对空集出 NaN
        for c in range(k):
            mask = new_labels == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)

        shift = np.sqrt(((new_centers - centers) ** 2).sum())
        labels, centers = new_labels, new_centers
        if shift < tol:                         # 中心位移不足 0.1 米即视为收敛
            break
    return labels, centers


def silhouette_score(D, labels):
    """轮廓系数（课时 37 手写实现）。D 为点到点距离矩阵，由调用方算一次复用。"""
    uniq = np.unique(labels)
    if len(uniq) <= 1:
        return 0.0
    n = len(labels)
    vals = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = (labels == c)
            if other.any():
                b = min(b, D[i, other].mean())
        denom = max(a, b)
        vals[i] = (b - a) / denom if denom > 0 else 0.0
    return float(vals.mean())


def dunn_index(D, labels):
    """邓恩指数（课时 37）：簇间最小距离 / 簇内最大直径。"""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    max_intra = 0.0
    for c in uniq:                              # 各簇最大直径
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            max_intra = max(max_intra, D[np.ix_(idx, idx)].max())
    min_inter = np.inf
    for i in range(len(uniq)):                  # 簇间最小距离
        for j in range(i + 1, len(uniq)):
            a = np.where(labels == uniq[i])[0]
            b = np.where(labels == uniq[j])[0]
            min_inter = min(min_inter, D[np.ix_(a, b)].min())
    if max_intra <= 0:
        return 0.0
    return float(min_inter / max_intra)


def choose_best_k(X):
    """遍历 K=K_MIN~K_MAX，用 score = 轮廓系数 + DUNN_WEIGHT × 邓恩指数 选最优（课时 37）。

    返回 (labels, best_k, sse)。sse 是最优划分下的**簇内平方误差和**（米²）：
    纪要 day1 写的聚类评价是三件套「SSE / 邓恩指数 / 轮廓系数」，SSE 只报不选——
    它随 K 单调下降，拿它当选择函数会永远选到 K_MAX；选 K 仍按课时 37 的 score。
    """
    n = len(X)
    if n < config.K_MIN:                        # 点太少，直接单簇
        return np.zeros(n, dtype=int), 1, 0.0
    if len(np.unique(X, axis=0)) == 1:          # 所有点同一个坐标：没有任何结构可分
        return np.zeros(n, dtype=int), 1, 0.0
    D = pairwise_euclidean(X)                   # 距离矩阵只算一次，遍历 k 时复用
    # 基线是「不拆分」，得分记 0：轮廓系数为 0 正好表示簇内簇间没有区别，
    # 只有真的比 0 好才值得拆。原先基线取 -inf，两点的退化拆法（得分恰好 0）
    # 也会赢过基线，把同一个位置的点硬分成两簇。
    best_score, best_k, best_labels, best_sse = 0.0, 1, np.zeros(n, dtype=int), 0.0
    for k in range(config.K_MIN, min(config.K_MAX, n) + 1):
        labels, centers = kmeans(X, k)
        if len(np.unique(labels)) < 2:          # 退化成单簇不给高分
            score = -np.inf
        else:
            score = silhouette_score(D, labels) + config.DUNN_WEIGHT * dunn_index(D, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
            best_sse = float(((X - centers[labels]) ** 2).sum())
    return best_labels, best_k, best_sse


def stage2_analysis(df=None):
    """逐用户手写 K-Means，保存 stage2_cluster.csv 并返回带簇标签的出行点表。"""
    log_step(3, "阶段2：逐用户手写 K-Means 聚类")
    if df is None:
        df = read_csv(config.STAGE1_TRAVEL_POINTS_FILE,
                      dtype={"imsi": str, "lac_id": str, "cell_id": str})
    df = df.copy()
    for c in ("lon", "lat"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log(f"读入阶段1出行点: {len(df):,} 行, {df['imsi'].nunique()} 个用户")

    np.random.seed(config.RANDOM_STATE)         # 课时 36：固定种子，簇结果可复现
    out_parts, chosen_ks, sses = [], [], []
    for imsi, g in df.groupby("imsi"):
        # 先投影成米再聚类：欧氏距离直接吃经纬度会把地图横向压扁 25%
        X = project_to_meters(g["lon"], g["lat"])
        labels, best_k, sse = choose_best_k(X)

        g = g.copy()
        g["cluster"] = labels
        size_map = pd.Series(labels).value_counts()
        g["is_single_cluster"] = g["cluster"].map(lambda c: 1 if size_map.get(c, 0) == 1 else 0)
        out_parts.append(g)
        chosen_ks.append(best_k)
        sses.append(sse)

    result = pd.concat(out_parts, ignore_index=True)
    n_clusters = result.groupby(["imsi", "cluster"]).ngroups
    n_single = int((result["is_single_cluster"] == 1).sum())
    ks = pd.Series(chosen_ks)
    log(f"共形成 {n_clusters} 个簇（其中单点簇 {n_single} 个），"
        f"最优K分布: 最小{int(ks.min())} / 最大{int(ks.max())} / 平均{ks.mean():.1f}")
    s = pd.Series(sses)
    # 纪要 day1 的三件套：SSE 与邓恩、轮廓一起报告。SSE 对每个用户各自最优划分求和，
    # 单位是米²，跨用户不可直接比（点数不同），所以看中位数而不是看总量。
    log(f"聚类质量（逐用户最优划分）：SSE 中位 {s.median():,.0f} 米² / "
        f"均值 {s.mean():,.0f} 米² / 最大 {s.max():,.0f} 米²，"
        f"折算均方根簇内半径约 {np.sqrt(s.median()):,.0f} 米")

    save_csv(result, config.STAGE2_CLUSTER_FILE)
    return result


if __name__ == "__main__":
    # 课时 36 §五：先用两团人造点验证手写实现本身
    np.random.seed(0)
    toy = np.array([[0, 0], [1, 0], [0, 1], [10, 0], [11, 0], [10, 1]], dtype=float)
    toy_labels, toy_centers = kmeans(toy, 2)
    log(f"kmeans 冒烟测试 labels={toy_labels.tolist()} "
        f"centers={np.round(toy_centers, 2).tolist()}")
    assert len(set(toy_labels.tolist())) == 2

    with Timer("阶段2KMeans聚类"):
        out = stage2_analysis()
    log(f"结果预览：\n{out.head(3).to_string(index=False)}")
