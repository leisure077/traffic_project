# -*- coding: utf-8 -*-
"""
visualize.py —— 六张结果可视化（课时 55~56）

入口 `visualize_all()`：一次读五张结果表（阶段3标签 / 阶段2簇 / 目的 / 方式 / 热力），
生成六张必交 PNG，外加两张附加 PNG。

  1 fig_stage3_pie    出行/驻留占比      —— 一天里移动点和停留点各占多少
  2 fig_hour_line     24 小时同轴叠画    —— 出行/驻留随时间如何变化（含早晚高峰底纹）
  3 fig_purpose_bar   出行目的柱状       —— 停留都在干什么
  4 fig_mode_bar      出行方式柱状       —— 出行都怎么去
  5 fig_heatmap       商圈×24h 热力矩阵  —— 哪里、什么时间最拥挤（按绝对人次上色）
  6 fig_3d_traj       Top5 用户 3D 轨迹  —— 个体一天的时空轨迹长什么样

  + fig_stage2_clusters 某用户的簇划分 + 阶段3 判定 —— 六张必交图里唯一没覆盖的
    「阶段2 到底做了什么」，讲义未要求，作为附加图补上
  + fig_dashboard       上述 6 张拼成的 3×2 总览，答辩用一张图讲完

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
# 3D 轨迹的连线配色：五个人的线要能一眼分开，且不能和「红=出行、绿=驻留」撞色
USER_COLORS = ["#9467bd", "#ff7f0e", "#1f77b4", "#8c564b", "#17becf"]


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
    """图2：24 小时出行/驻留叠在同一张图上（对齐讲义示例的排版）。

    低频点改为保留后出行/驻留量级接近（本次驻留约为出行的 2.9 倍），共轴直接画，
    不再需要放大小格；早晚高峰用淡蓝底纹标注，只是标注、不参与判定。
    左右双轴（两条线各自一套刻度）不用：那会让两条线看起来可比，
    读者会按交叉点误读「什么时候出行超过了驻留」。
    """
    tmp = pd.DataFrame({"hour": _hour_of(label_df["time"]), "label": label_df["label"]}).dropna()
    hours = list(range(24))
    stay = tmp[tmp["label"] == 0].groupby("hour").size().reindex(hours, fill_value=0)
    travel = tmp[tmp["label"] == 1].groupby("hour").size().reindex(hours, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for start, end, name in config.PEAK_HOURS:         # 底纹只是标注，不参与判定
        ax.axvspan(start, end, color="#1f77b4", alpha=.10)
    ax.plot(hours, stay.values, marker="o", color=GREEN, label="驻留点")
    ax.plot(hours, travel.values, marker="^", color=RED, label="出行点")
    ax.set_xlabel("小时")
    ax.set_ylabel("点数")
    ax.set_xticks(hours)
    ax.set_ylim(0, stay.max() * 1.15)
    ax.legend(loc="upper left")
    ax.grid(alpha=.3)

    # 示例图在这里标的是「早高峰」，但我们这份数据的出行最高点在 23 时，
    # 早晚高峰底纹里出行并没有明显抬起来，照写「早高峰」就成了图上没有的结论，
    # 所以只标实测最高点；峰值落在傍晚以远时把文字放到左侧，避免画出坐标外。
    peak_hour = int(travel.idxmax())
    ax.annotate(f"出行最高 {peak_hour}时（{int(travel.max())} 点）",
                (peak_hour, travel.max()),
                xytext=(peak_hour - 6.5 if peak_hour >= 18 else peak_hour + 1.5,
                        travel.max() + stay.max() * .12),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                color=RED, fontsize=9)

    fig.suptitle(f"24 小时出行 / 驻留点分布（同一坐标叠画：驻留量级约为出行的 "
                 f"{stay.sum() / max(travel.sum(), 1):.0f} 倍）")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    """图5：Top10 商圈 × 24 小时热力矩阵，按绝对人次上色（对齐讲义示例）。

    窗口计数取该小时内的峰值：一小时内有 60 个十分钟窗口，取最大那个代表这一小时最挤的程度。
    色标用的是**真实人次**而不是每行各自的最大值。之前按行归一化能把每个商圈的日内起伏都
    看清，但代价是棋盘山（峰值 10）和中街（峰值 215）会画得和太原街（峰值 2570）一样红，
    读者按颜色估人数会高估十几倍。示例图同样是绝对刻度，这里跟过来更诚实；
    小商圈看不出的起伏，用每格里的数字标签补回。
    """
    t = pd.to_datetime(heat_df["time"])
    df = heat_df.assign(hour=t.dt.hour).dropna(subset=["hour"])
    pivot = df.groupby(["biz_name", "hour"])["count"].max().unstack(fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    top = pivot.max(axis=1).sort_values(ascending=False).head(HEATMAP_TOP)
    M = pivot.loc[top.index]

    values = M.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 1 + 1.1 * len(M)))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=values.max())
    ax.set_xticks(range(24), [f"{h}时" for h in range(24)], fontsize=8, rotation=60)
    ax.set_yticks(range(len(M)), M.index)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if v > 0:
                ax.text(j, i, int(v), ha="center", va="center", fontsize=6.5,
                        color="white" if v > values.max() * .6 else "#333333")
        ax.text(23.7, i, f"峰值 {int(row.max()):,}", va="center", fontsize=8)
    ax.set_xlim(-0.5, 26.5)
    ax.set_xlabel("小时")
    ax.set_title(f"商圈 × 24 小时人流热力（{config.HEATMAP_WINDOW_SIZE} 分钟窗口内的峰值人次）")
    fig.colorbar(im, ax=ax, label="窗口累计人次", pad=0.02)
    _save(fig, config.FIG_HEATMAP)


def plot_3d_traj(label_df):
    """图6：Top5 用户 3D 时空轨迹。x/y 是位置、z 是时间，驻留表现为垂直上升的柱。

    点的形状按标签区分（对齐示例图）：红三角=出行、绿圆点=驻留；
    连线只表示同一个用户的时间先后，用不同颜色分人。
    """
    df = label_df.copy()
    df["_dt"] = pd.to_datetime(df["time"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["_dt", "lon", "lat"])
    df["z"] = df["_dt"].dt.hour + df["_dt"].dt.minute / 60.0
    top_users = df["imsi"].value_counts().head(SCATTER_3D_TOP).index

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, u in enumerate(top_users):
        g = df[df["imsi"] == u].sort_values("_dt")
        ax.plot(g["lon"], g["lat"], g["z"], marker="", color=USER_COLORS[i % len(USER_COLORS)],
                alpha=.75, label=f"用户{str(u)[-4:]}轨迹")
        stay_g, travel_g = g[g["label"] == 0], g[g["label"] == 1]
        ax.scatter(stay_g["lon"], stay_g["lat"], stay_g["z"], marker="o", s=18,
                   color=GREEN, depthshade=False)
        ax.scatter(travel_g["lon"], travel_g["lat"], travel_g["z"], marker="^", s=26,
                   color=RED, depthshade=False)
    # 图例先放两类点（示例图就是这个读法），再放各用户连线
    ax.scatter([], [], marker="o", s=18, color=GREEN, label="驻留点")
    ax.scatter([], [], marker="^", s=26, color=RED, label="出行点")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_zlabel("时间(小时)")
    ax.set_title(f"Top{len(top_users)} 用户 3D 时空轨迹（红三角=出行 / 绿圆点=驻留）")
    ax.legend(fontsize=8)
    ax.view_init(elev=25, azim=50)
    _save(fig, config.FIG_3D_TRAJ)


def plot_stage2_clusters(cluster_df, label_df):
    """附加图：单个用户的阶段2 簇划分 + 阶段3 判定结果。

    六张必交图里没有一张能说明「阶段2 到底做了什么」——它既不是占比也不是分布。
    这里挑一个簇够多、且阶段3 判定**两种结果都有**的用户，把点按簇上色、簇心画黑叉，
    图例直接标出该簇最后被判成驻留还是出行，一条「分组 → 精判」的链在一张图上讲完。
    只挑全判出行的用户会让图例变成六行一样的字，看不出精判纠正了什么。
    左格用灰色背景点说明阶段1 的漏斗只放行了少量疑似出行点，右格放大看簇的空间结构。
    """
    pts = cluster_df.merge(label_df[["imsi", "time", "label"]],
                           on=["imsi", "time"], how="left")          # (imsi,time) 唯一
    per_cluster = (pts.groupby(["imsi", "cluster"])
                   .agg(n=("label", "size"), travel=("label", "sum")).reset_index())
    per_cluster["verdict"] = np.where(per_cluster["travel"] == 0, "驻留", "出行")
    # 同一基站的点经纬度完全相同，簇会缩成一个可见点，所以要挑真正散开的用户。
    # 两个门槛都换算成公里（经度 1° 要乘 cos(纬度)，否则东西向被高估 25%）：
    #   fill_km > 1  —— 窄方向也得铺到 1 公里以上，图才不是挤在一团；
    #   ratio > 0.3  —— 宽窄方向的比值。少了这条会挑中南北拉 28 公里、东西只有 3 公里的
    #                  用户，配上等比例坐标轴就退化成竖直一条带，看不出簇的空间分布。
    def span_km(col, deg_km):
        span = pts.groupby("imsi")[col].agg(lambda s: float(s.max() - s.min()))
        if col == "lon":
            span = span * np.cos(np.radians(pts.groupby("imsi")["lat"].mean()))
        return span * deg_km

    lon_km, lat_km = span_km("lon", 111.32), span_km("lat", 110.57)
    shape = pd.DataFrame({"km_lon": lon_km, "km_lat": lat_km})
    shape["fill_km"] = shape.min(axis=1)
    shape["ratio"] = shape["fill_km"] / shape.max(axis=1)
    cand = (per_cluster.groupby("imsi")
            .agg(clusters=("cluster", "size"), kinds=("verdict", "nunique"), n=("n", "sum"))
            .join(shape)
            .query("clusters >= 3 and kinds == 2 and fill_km > 1.0 and ratio > 0.3")
            .sort_values(["clusters", "n"], ascending=False))
    if len(cand) == 0:
        raise ValueError("没有「簇数>=3、驻留/出行判定都有、且两向都铺开」的用户，跳过簇划分图")
    user = cand.index[0]

    g = pts[pts["imsi"] == user]
    all_pts = label_df[label_df["imsi"] == user]
    bg = all_pts[~all_pts["time"].isin(set(g["time"]))]
    clusters = [(int(c), g[g["cluster"] == c]) for c in sorted(g["cluster"].unique(), key=int)]
    lat_mid = float(g["lat"].mean())
    aspect = 1.0 / np.cos(np.radians(lat_mid))     # 图上 1 厘米东西/南北代表同样米数

    # 左格看全貌、右格看结构：只画放大图会看不出「一天 340 个点里只有 19 个进了聚类」
    # 这个漏斗量级；只画全貌图，4 个簇又缩成角落里的几个点，看不出簇的空间含义。
    fig, (wide, zoom) = plt.subplots(1, 2, figsize=(14, 7))
    wide.scatter(bg["lon"], bg["lat"], s=6, color="#cccccc", alpha=.6, zorder=1)
    for i, (c, gc) in enumerate(clusters):
        wide.scatter(gc["lon"], gc["lat"], s=26, color=plt.cm.tab10(i % 10), zorder=2)
    wide.set_aspect(aspect)
    wide.set_title(f"全日视角：{len(all_pts):,} 个点里只有 {len(g)} 个进入聚类\n"
                   f"（灰点＝阶段1 未放行的驻留点，默认判驻留）")

    for i, (c, gc) in enumerate(clusters):
        verdict = "驻留" if int(gc["label"].sum()) == 0 else "出行"
        zoom.scatter(gc["lon"].mean(), gc["lat"].mean(), marker="X", s=150,
                     color="black", zorder=2)
        zoom.scatter(gc["lon"], gc["lat"], s=48, color=plt.cm.tab10(i % 10), zorder=3,
                     edgecolor="white", linewidth=.5,
                     label=f"簇{c}（{len(gc)} 点 · {verdict}）")
    zoom.scatter([], [], marker="X", s=150, color="black", label="簇心")
    zoom.set_aspect(aspect)
    zoom.set_title("放大：按簇上色 + 簇心（黑叉），图例是阶段3 的最终判定")
    zoom.legend(fontsize=8, loc="best", framealpha=.95)

    for ax in (wide, zoom):
        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
    fig.suptitle(f"阶段2 簇划分 + 阶段3 判定（用户 …{str(user)[-4:]}，"
                 f"{len(g)} 个疑似出行点分成 {len(clusters)} 簇）")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, config.FIG_CLUSTER)


def build_dashboard():
    """附加图：把 6 张结果图拼成 3×2 总览，答辩时一张图讲完全链。"""
    from PIL import Image, ImageDraw, ImageFont

    tiles = []
    for path in config.RESULT_FIGS:
        img = Image.open(path).convert("RGB")
        h = 620
        tiles.append(img.resize((int(img.width * h / img.height), h)))
    cols, rows, gap, header = 3, 2, 16, 64
    cell_w, cell_h = max(t.width for t in tiles), max(t.height for t in tiles)
    width = cell_w * cols + gap * (cols + 1)
    height = cell_h * rows + gap * (rows + 1) + header
    board = Image.new("RGB", (width, height), "white")
    for i, t in enumerate(tiles):
        # 各图宽高比不同，按格子里居中贴，否则同一行的图会挤在左边、右边留大片空白
        x = gap + (i % cols) * (cell_w + gap) + (cell_w - t.width) // 2
        y = gap + header + (i // cols) * (cell_h + gap)
        board.paste(t, (x, y))

    # PIL 的默认字体不含中文，直接 draw.text 会画成一串方框，必须显式加载中文字体
    font = None
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                 "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        try:
            font = ImageFont.truetype(path, 34)
            break
        except OSError:
            continue
    title = "交通时空数据分析 · 结果总览"
    draw = ImageDraw.Draw(board)
    if font is None:
        draw.text((gap, 18), title, fill="black")
    else:
        tw = draw.textlength(title, font=font)
        draw.text(((width - tw) / 2.0, (header - font.size) / 2.0), title,
                  fill="black", font=font)
    board.save(config.FIG_DASHBOARD)
    log(f"  图已保存 {config.FIG_DASHBOARD}")


def visualize_all():
    """读四张结果表，生成六张必交图 + 两张附加图；单张图失败只警告，不影响其他图。"""
    log_step(8, "可视化（6 张图 + 2 张附加图）")
    setup_zh()

    label_df = read_csv(config.STAGE3_LABEL_FILE,
                        dtype={"imsi": str, "time": str, "lac_id": str, "cell_id": str})
    label_df["label"] = pd.to_numeric(label_df["label"], errors="coerce")
    label_df["lon"] = pd.to_numeric(label_df["lon"], errors="coerce")
    label_df["lat"] = pd.to_numeric(label_df["lat"], errors="coerce")
    purpose_df = read_csv(config.PURPOSE_RESULT_FILE, dtype={"imsi": str})
    mode_df = read_csv(config.MODE_RESULT_FILE, dtype={"imsi": str})
    heat_df = read_csv(config.HEATMAP_RESULT_FILE)
    heat_df["count"] = pd.to_numeric(heat_df["count"], errors="coerce")
    cluster_df = read_csv(config.STAGE2_CLUSTER_FILE, dtype={"imsi": str, "time": str})
    cluster_df["cluster"] = pd.to_numeric(cluster_df["cluster"], errors="coerce")

    jobs = [("出行/驻留饼图", plot_stage3_pie, (label_df,)),
            ("24小时双折线", plot_hour_line, (label_df,)),
            ("出行目的柱状", plot_purpose_bar, (purpose_df,)),
            ("出行方式柱状", plot_mode_bar, (mode_df,)),
            ("商圈热力矩阵", plot_heatmap, (heat_df,)),
            ("Top5 3D 轨迹", plot_3d_traj, (label_df,)),
            ("阶段2簇划分", plot_stage2_clusters, (cluster_df, label_df))]
    for name, fn, args in jobs:
        try:
            if any(len(a) == 0 for a in args):
                raise ValueError("输入结果表为空")
            fn(*args)
        except Exception as e:                       # 一张图坏了不拖累其余五张
            log(f"{name} 生成失败: {e}", level="WARN")
    try:
        build_dashboard()
    except Exception as e:
        log(f"总览拼图 生成失败: {e}", level="WARN")
    return config.RESULT_FIGS


if __name__ == "__main__":
    with Timer("可视化"):
        visualize_all()
