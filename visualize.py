# -*- coding: utf-8 -*-
"""
visualize.py —— 六张结果可视化（课时 55~56）

入口 `visualize_all()`：一次读四张结果表（阶段3标签 / 目的 / 方式 / 热力），生成六张 PNG。

  1 fig_stage3_pie    出行/驻留占比      —— 一天里移动点和停留点各占多少
  2 fig_hour_line     24 小时双折线      —— 出行/驻留随时间如何变化
  3 fig_purpose_bar   出行目的柱状       —— 停留都在干什么
  4 fig_mode_bar      出行方式柱状       —— 出行都怎么去
  5 fig_heatmap       商圈×24h 热力矩阵  —— 哪里、什么时间最拥挤
  6 fig_3d_traj       Top5 用户 3D 轨迹  —— 个体一天的时空轨迹长什么样

两个反复用到的画图规矩：
  · 固定类别顺序 + reindex 补 0：某类为 0 时仍保留它的位置，柱子不会整体错位；
    颜色与类别绑定（居住蓝/购物橙/旅游紫、地铁红/公交蓝/小汽车橙、其他一律灰），换数据不串色；
  · 单张图失败只 WARN、不影响其他图，每张图画完立即 plt.close(fig) 释放内存。

`python visualize.py` 单独运行本模块（需先跑完前七个模块）。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from utils import log, log_step, read_csv, setup_zh, Timer

GREEN, RED, BLUE = "#2ca02c", "#d62728", "#1f77b4"
ORANGE, PURPLE, GRAY = "#ff7f0e", "#9467bd", "gray"
PURPOSE_COLORS = {"居住": BLUE, "购物": ORANGE, "旅游": PURPLE, "其他": GRAY}
MODE_COLORS = {"地铁": RED, "公交": BLUE, "小汽车": ORANGE, "其他": GRAY}
SCATTER_3D_TOP = 5          # 3D 轨迹画点数最多的前 5 个用户
HEATMAP_TOP = 10            # 热力矩阵取峰值最高的前 10 个商圈


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    log(f"  图已保存 {path}")


def _hour_of(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S", errors="coerce").dt.hour


def plot_stage3_pie(label_df):
    """图1：驻留/出行占比饼图。"""
    stay = int((label_df["label"] == 0).sum())
    travel = int((label_df["label"] == 1).sum())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([stay, travel], labels=["驻留", "出行"], autopct="%1.1f%%",
           colors=[GREEN, RED], startangle=90, explode=[0, 0.06])
    ax.set_title(f"阶段3：出行/驻留占比（共 {stay + travel:,} 点）")
    _save(fig, config.FIG_STAGE3_PIE)


def plot_hour_line(label_df):
    """图2：24 小时上下两个子图。reindex(range(24)) 保证没有点的小时也显示为 0。

    驻留量级是出行的二十多倍，放进同一根轴出行会被压成贴底线；
    用左右双轴并排则两条线看起来可比，读者会按交叉点误读「什么时候出行超过驻留」。
    所以拆成上下两格、各自量级、共用横轴：形状可比，高度不可比。
    """
    tmp = pd.DataFrame({"hour": _hour_of(label_df["time"]), "label": label_df["label"]}).dropna()
    hours = list(range(24))
    stay = tmp[tmp["label"] == 0].groupby("hour").size().reindex(hours, fill_value=0)
    travel = tmp[tmp["label"] == 1].groupby("hour").size().reindex(hours, fill_value=0)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax.plot(hours, stay.values, marker="o", color=GREEN, label="驻留点")
    ax.fill_between(hours, stay.values, alpha=.12, color=GREEN)
    ax.set_ylabel("驻留点数")
    ax.legend(loc="upper left")
    ax.grid(alpha=.3)

    ax2.plot(hours, travel.values, marker="s", color=RED, label="出行点")
    ax2.fill_between(hours, travel.values, alpha=.12, color=RED)
    ax2.set_ylabel("出行点数")
    ax2.set_xlabel("小时")
    ax2.set_xticks(hours)
    ax2.legend(loc="upper left")
    ax2.grid(alpha=.3)

    fig.suptitle("24 小时出行 / 驻留分布（上下两格各自量级，只比形状不比高低）")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, config.FIG_HOUR_LINE)


def _plot_cat_bar(counts, order, colors, path, ylabel, title, total):
    """固定顺序 + 缺失补 0 + 类别绑定配色，柱顶标「数量 + 占比」。"""
    values = [int(counts.get(c, 0)) for c in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(order, values, color=[colors[c] for c in order], edgecolor="black", alpha=.9)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * .02, f"{v}\n({v / total * 100:.1f}%)", ha="center", fontsize=9)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)


def _ordered_counts(series, fixed_order):
    """固定类别在前，静态表里新冒出来的类别插在「其他」之前，保证「其他」永远收尾。"""
    vc = series.value_counts()
    extra = sorted(c for c in vc.index if c not in fixed_order and c != "其他")
    return fixed_order[:-1] + extra + ["其他"], vc


def plot_purpose_bar(purpose_df):
    """图3：出行目的柱状。"""
    order, vc = _ordered_counts(purpose_df["final_purpose"], config.PURPOSE_ORDER)
    colors = {c: PURPOSE_COLORS.get(c, GRAY) for c in order}
    _plot_cat_bar(vc, order, colors, config.FIG_PURPOSE_BAR,
                  "驻留段数", f"出行目的分布（{len(purpose_df)} 段）", len(purpose_df))


def plot_mode_bar(mode_df):
    """图4：出行方式柱状。"""
    order, vc = _ordered_counts(mode_df["final_mode"], config.MODE_ORDER)
    colors = {c: MODE_COLORS.get(c, GRAY) for c in order}
    _plot_cat_bar(vc, order, colors, config.FIG_MODE_BAR,
                  "出行段数", f"出行方式分布（{len(mode_df)} 段）", len(mode_df))


def plot_heatmap(heat_df):
    """图5：Top10 商圈 × 24 小时热力矩阵。窗口计数取该小时内的峰值。

    色标按行各自归一化：四个商圈量级差到两三个数量级（太原街峰值 2570、棋盘山只有 10），
    共用一套色标会把小的那几行整体冲成同一色，日内起伏完全看不出来。
    每行右侧标出该行真实峰值，绝对量级信息不丢。
    """
    t = pd.to_datetime(heat_df["time"])
    df = heat_df.assign(hour=t.dt.hour).dropna(subset=["hour"])
    pivot = df.groupby(["biz_name", "hour"])["count"].max().unstack(fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    top = pivot.max(axis=1).sort_values(ascending=False).head(HEATMAP_TOP)
    M = pivot.loc[top.index]

    values = M.to_numpy(dtype=float)
    row_max = values.max(axis=1)
    rel = values / np.maximum(row_max, 1)[:, None]        # 全零行保持 0，不出 NaN

    fig, ax = plt.subplots(figsize=(11, 1 + 1.1 * len(M)))
    im = ax.imshow(rel, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(24), [f"{h}时" for h in range(24)], fontsize=8, rotation=60)
    ax.set_yticks(range(len(M)), M.index)
    for i, v in enumerate(row_max):
        ax.text(23.7, i, f"峰值 {int(v)}", va="center", fontsize=8)
    ax.set_xlim(-0.5, 26.5)
    ax.set_xlabel("小时")
    ax.set_title(f"商圈 × 24 小时人流热力（{config.HEATMAP_WINDOW_SIZE} 分钟窗口峰值，色标按行归一化）")
    fig.colorbar(im, ax=ax, label="占本商圈当日峰值比例", pad=0.02)
    _save(fig, config.FIG_HEATMAP)


def plot_3d_traj(label_df):
    """图6：Top5 用户 3D 时空轨迹。x/y 是位置、z 是时间，驻留表现为垂直上升的柱。"""
    df = label_df.copy()
    df["_dt"] = pd.to_datetime(df["time"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["_dt", "lon", "lat"])
    top_users = df["imsi"].value_counts().head(SCATTER_3D_TOP).index

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, u in enumerate(top_users):
        g = df[df["imsi"] == u].sort_values("_dt")
        z = g["_dt"].dt.hour + g["_dt"].dt.minute / 60.0
        ax.plot(g["lon"], g["lat"], z, marker="o", markersize=3,
                color=plt.cm.tab10(i % 10), label=f"用户{str(u)[-4:]}")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_zlabel("时间(小时)")
    ax.set_title(f"Top{len(top_users)} 用户 3D 时空轨迹")
    ax.legend(fontsize=8)
    ax.view_init(elev=25, azim=50)
    _save(fig, config.FIG_3D_TRAJ)


def visualize_all():
    """读四张结果表，生成六张图；单张图失败只警告，不影响其他图。"""
    log_step(8, "可视化（6 张图）")
    setup_zh()

    label_df = read_csv(config.STAGE3_LABEL_FILE,
                        dtype={"imsi": str, "lac_id": str, "cell_id": str})
    label_df["label"] = pd.to_numeric(label_df["label"], errors="coerce")
    label_df["lon"] = pd.to_numeric(label_df["lon"], errors="coerce")
    label_df["lat"] = pd.to_numeric(label_df["lat"], errors="coerce")
    purpose_df = read_csv(config.PURPOSE_RESULT_FILE, dtype={"imsi": str})
    mode_df = read_csv(config.MODE_RESULT_FILE, dtype={"imsi": str})
    heat_df = read_csv(config.HEATMAP_RESULT_FILE)
    heat_df["count"] = pd.to_numeric(heat_df["count"], errors="coerce")

    jobs = [("出行/驻留饼图", plot_stage3_pie, label_df),
            ("24小时双折线", plot_hour_line, label_df),
            ("出行目的柱状", plot_purpose_bar, purpose_df),
            ("出行方式柱状", plot_mode_bar, mode_df),
            ("商圈热力矩阵", plot_heatmap, heat_df),
            ("Top5 3D 轨迹", plot_3d_traj, label_df)]
    for name, fn, data in jobs:
        try:
            if len(data) == 0:
                raise ValueError("结果表为空")
            fn(data)
        except Exception as e:                       # 一张图坏了不拖累其余五张
            log(f"{name} 生成失败: {e}", level="WARN")
    return config.RESULT_FIGS


if __name__ == "__main__":
    with Timer("可视化"):
        visualize_all()
