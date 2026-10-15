# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具（课时 13 手写 Haversine、课时 14 utils 模块实战）

包含：向量化 Haversine、时间解析/格式化、CSV 读写（无表头/GBK/utf-8-sig）、
带级别与时间戳的日志、步骤标题、DataFrame 概览、Timer 上下文管理器、
速度换算 kmh、夜间判定 is_night、matplotlib 中文字体配置。

`python utils.py` 跑三项自测（课时 14 §三）。
"""
import os
import time
import datetime

import numpy as np
import pandas as pd

import config

EARTH_R_KM = 6371.0088


# ----------------------------------------------------------------------
# 距离
# ----------------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2, unit="km"):
    """向量化球面距离。支持标量与 numpy 广播（M,1）×（1,N）→（M,N）。

    课时 13：聚类在经纬度平面上用欧氏距离近似，但所有和米级阈值比较的距离都用这个真实值。
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))
    return km * 1000.0 if unit == "m" else km


def pairwise_euclidean(X):
    """点集自身的欧氏距离矩阵（课时 37 轮廓/邓恩指数的输入，经纬度平面近似）。"""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def kmh(dist_km, seconds):
    """千米 + 秒 → km/h。时间非正时返回 NaN，避免 inf 污染均值。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.asarray(dist_km, dtype=float) * 3600.0 / np.asarray(seconds, dtype=float)
    return np.where(np.asarray(seconds, dtype=float) > 0, spd, np.nan)


# ----------------------------------------------------------------------
# 时间
# ----------------------------------------------------------------------
def parse_time(series, fmt="%Y%m%d%H%M%S"):
    """'20181003081530' 字符串列 → datetime 列，解析失败给 NaT 而不是报错。"""
    return pd.to_datetime(series.astype(str), format=fmt, errors="coerce")


def format_time(ts, fmt="%Y%m%d%H%M%S"):
    """datetime → 字符串；NaT/None 原样返回空串。"""
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).strftime(fmt)


def is_night(hour):
    """课时 44：夜间时段 0-6 或 20-23。居住判据需要段端点落在夜间。"""
    return ((hour >= config.NIGHT_HOURS[0][0]) & (hour <= config.NIGHT_HOURS[0][1])) | \
           ((hour >= config.NIGHT_HOURS[1][0]) & (hour <= config.NIGHT_HOURS[1][1]))


# ----------------------------------------------------------------------
# CSV 读写
# ----------------------------------------------------------------------
def read_csv(path, **kw):
    """统一读 CSV。默认 dtype=str 防止 imsi 这类长数字串被转成 float 丢精度。"""
    kw.setdefault("dtype", str)
    kw.setdefault("encoding", "utf-8")
    return pd.read_csv(path, **kw)


def read_csv_gbk(path, **kw):
    """读 GBK 编码的静态数据（课时 16：两个 POI 表是 GBK，其余是 UTF-8）。"""
    kw.setdefault("encoding", "gbk")
    return read_csv(path, **kw)


def save_csv(df, path, encoding="utf-8-sig"):
    """写 CSV，utf-8-sig 保证 Excel 直接双击打开中文不乱码。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding=encoding)
    log(f"已保存 {path}（{len(df):,} 行）")
    return path


def drop_out_of_range_poi(df, name="POI"):
    """剔掉经纬度不在沈阳市域内的静态表行，并打印剔了谁（课时 51 的脏数据防御）。

    真实案例：商业区数据.csv 第 23 行沈阳北站 lon=23.454567，经度少了前导的 1。
    它这次恰好匹配不上任何点、没污染结果，但脏点留在候选里等于让 argmin 白算一列
    距离，换一批数据就可能把某个点整条拽飞，所以读表时就剔掉并留痕。
    注意上界用 config.POI_LAT_RANGE=43.5 而不是信令那套 41.1~41.9：
    目的/方式静态表里 lat≈42.7 的三十来行是康平县的真实 POI，窄区间会误删。
    """
    lo_lon, hi_lon = config.POI_LNG_RANGE
    lo_lat, hi_lat = config.POI_LAT_RANGE
    bad = ~(df["lon"].between(lo_lon, hi_lon) & df["lat"].between(lo_lat, hi_lat))
    if bad.any():
        label_cols = [c for c in ("biz_name", "poi_name", "station_name", "station", "name")
                      if c in df.columns]
        detail = "；".join(
            f"lon={df.at[i, 'lon']:.6f} {df.at[i, label_cols[0]] if label_cols else ''}"
            for i in df.index[bad][:5])
        log(f"{name} 剔范围外坐标 {int(bad.sum())} 行（{lo_lon}~{hi_lon}, {lo_lat}~{hi_lat}）："
            f"{detail}", level="WARN")
    return df[~bad].reset_index(drop=True)


# ----------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------
def log(msg, level="INFO"):
    """带时间戳和级别的日志（课时 14）。"""
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] [{level:<5}] {msg}", flush=True)


def log_step(step, title):
    print("=" * 66, flush=True)
    log(f"步骤 {step}：{title}", level="STEP")
    print("=" * 66, flush=True)


def print_df_stats(df, name=""):
    """课时 14：快速看一个 DataFrame 的规模、列、缺失和数值分布。"""
    print(f"--- DataFrame {name}: {len(df)} 行 × {df.shape[1]} 列 ---")
    print(f"列名: {list(df.columns)}")
    na = df.isna().sum()
    na = na[na > 0]
    print(f"缺失: {na.to_dict()}" if len(na) else "缺失: 无")
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        print(num.describe().loc[["mean", "min", "50%", "max"]].round(3).to_string())


class Timer:
    """课时 14/57：`with Timer("数据清洗"):` 自动打印耗时。可嵌套复用。"""

    def __init__(self, name=""):
        self.name = name or "代码块"
        self.elapsed = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._t0
        log(f"[计时] {self.name} 耗时 {self.elapsed:.2f} 秒")
        return False


# ----------------------------------------------------------------------
# 绘图样式（课时 55）
# ----------------------------------------------------------------------
def setup_zh():
    """中文字体、负号、无界面后端、统一 dpi。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14


# ----------------------------------------------------------------------
# 自测（课时 14 §三）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    log_step(0, "utils.py 自测")

    # 1) Haversine：沈阳站→沈阳北站实际约 5km，与已知值比对
    d = haversine(123.405, 41.795, 123.433, 41.811, unit="m")
    print(f"[1] haversine(123.405,41.795 → 123.433,41.811) = {d:,.1f} m")
    assert 2000 < d < 7000, "沈阳城区两点距离应落在 2~7km"
    assert abs(haversine(0, 0, 0, 0)) < 1e-9, "同点距离应为 0"
    # 广播形状：(M,1)×(1,N) → (M,N)
    lon = np.array([[123.40], [123.45]])
    lat = np.array([[41.80], [41.82]])
    dm = haversine(lon, lat, np.array([[123.42, 123.50]]), np.array([[41.81, 41.85]]), unit="m")
    assert dm.shape == (2, 2), f"广播结果形状应为 (2,2)，实为 {dm.shape}"
    # 广播的每一格必须等于对应标量调用，否则说明维度被静默转置了
    assert abs(dm[1, 1] - haversine(123.45, 41.82, 123.50, 41.85, unit="m")) < 1e-6
    print(f"    广播 (2,1)×(1,2) → {dm.shape}，逐格与标量实现一致")

    # 2) 时间解析 + 速度换算
    t = parse_time(pd.Series(["20181003081530", "20181003081630", "bad"]))
    print(f"[2] parse_time → {t.dt.strftime('%H:%M:%S').tolist()}")
    assert str(t.iloc[0]) == "2018-10-03 08:15:30" and pd.isna(t.iloc[2]), "解析含容错"
    secs = np.diff(t[:2].to_numpy()).astype("timedelta64[s]").astype(float)
    print(f"    kmh(1.0km, {secs[0]:.0f}s) = {kmh(1.0, secs[0]):.1f} km/h；kmh(x, 0s) = {kmh(1.0, 0.0)}")
    assert abs(kmh(1.0, 60.0) - 60.0) < 1e-9 and np.isnan(kmh(1.0, 0.0))

    # 3) 夜间判定与日志级别
    print(f"[3] is_night: 3时={is_night(3)} 12时={is_night(12)} 22时={is_night(22)}")
    assert is_night(3) and is_night(22) and not is_night(12)
    with Timer("自测计时"):
        time.sleep(0.05)

    have, missing = config.check_data_files()
    print(f"    输入数据 {len(have)}/6 就位" + (f"，缺失: {missing}" if missing else ""))
    print("utils 自测全部通过")
