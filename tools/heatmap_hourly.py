# -*- coding: utf-8 -*-
"""
tools/heatmap_hourly.py —— 图5 的另一版热力图：按「每小时命中人次」出图（可选工具）

为什么要另一版：`visualize.plot_heatmap()` 画的是 10 分钟滑动窗口在小时内的峰值
（太原街 23 时 = 2,570），老师示例同一格是 32,187，量级差 12 倍——不是数据差那么多，
是两张图画的本就不是一个量。这里改画该小时的真实命中人次（太原街 23 时 = 12,502），
与示例同为万级，形状对得上（5 时谷底、17 时后拉起、23 时最高，归一化后与示例 r=0.946）。

数值必须在滑动窗口之前统计：heatmap_result.csv 存的是 60 个互相重叠的窗口，
按小时求和会把每个人数 10 次。所以这里直接复用 heatmap.py 的读入与商圈匹配，
在 rolling 之前按 (小时, 商圈) 计数。匹配口径一字未改：最近商圈、1000m 内才算命中。

同时去掉原图右侧的「峰值 X」文字列和它撑出来的空白，数字只写深色格（示例做法），
行只留「商圈」两类（北站是交通枢纽、棋盘山/世博园是景区，与示例的两行同类别）。

跑法：`python tools/heatmap_hourly.py` → result/figures/fig_heatmap_hourly.png
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                      # noqa: E402
import heatmap                     # noqa: E402
from utils import log, save_csv, setup_zh, Timer  # noqa: E402

HOURLY_FILE = os.path.join(config.RESULT_DIR, "heat_hourly.csv")
FIG_FILE = os.path.join(config.FIG_DIR, "fig_heatmap_hourly.png")
LABEL_RATIO = 0.5                 # 只在 >= 全表最大值一半的格子里写数字


def hourly_counts():
    """复用热力模块的读入+商圈匹配，在 10 分钟滚动之前按 (商圈, 小时) 计数。

    只保留名字以「商圈」结尾的行：沈阳北站是交通枢纽、棋盘山风景区和沈阳世博园是景区，
    不属于商圈，与示例图5 的两行（太原街商圈、中街商圈）同一类别。
    """
    stream = heatmap.match_business(heatmap.load_stream(), heatmap.load_business())
    stream = stream[stream["biz_name"].str.endswith("商圈")]
    stream["hour"] = stream["time_dt"].dt.hour
    pivot = stream.groupby(["biz_name", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    pivot["全天合计"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("全天合计", ascending=False)
    log(f"小时命中表：{pivot.shape[0]} 个商圈 × 24 小时，全天合计 {int(pivot['全天合计'].sum()):,} 人次")
    return pivot


def plot(pivot):
    M = pivot.drop(columns="全天合计")
    values = M.to_numpy(dtype=float)
    vmax = values.max()

    setup_zh()
    fig, ax = plt.subplots(figsize=(11.5, 1.45 * len(M) + 1.2))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(24), [f"{h}时" for h in range(24)], fontsize=8, rotation=60)
    ax.set_yticks(range(len(M)), M.index)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if v >= vmax * LABEL_RATIO:
                ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(len(M) - 0.5, -0.5)
    ax.set_xlabel("小时")
    ax.set_ylabel("商圈")
    ax.set_title(f"商圈 × 24 小时人流热力（每小时命中人次，商圈 {config.HEATMAP_MATCH_THRESHOLD}m 内）")
    fig.colorbar(im, ax=ax, label="小时命中人次", pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_FILE)
    plt.close(fig)
    log(f"已出图：{FIG_FILE}")


if __name__ == "__main__":
    with Timer("小时级热力图"):
        p = hourly_counts()
        save_csv(p, HOURLY_FILE)
        plot(p)
