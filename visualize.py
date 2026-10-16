# -*- coding: utf-8 -*-
"""
visualize.py —— 六张结果可视化（课时 55~56）

入口 `visualize_all()`：一次读五张结果表（阶段3标签 / 阶段2簇 / 目的 / 方式 / 小时命中表），
生成六张必交 PNG，外加三张附加 PNG。

  1 fig_stage3_pie    出行/驻留占比      —— 一天里移动点和停留点各占多少
  2 fig_hour_line     24 小时同轴叠画    —— 出行/驻留随时间如何变化（含早晚高峰底纹）
  3 fig_purpose_bar   出行目的柱状       —— 停留都在干什么
  4 fig_mode_bar      出行方式柱状       —— 出行都怎么去
  5 fig_heatmap       商圈×24h 热力矩阵  —— 哪里、什么时间最拥挤（按绝对人次上色）
  6 fig_3d_traj       Top5 用户 3D 轨迹  —— 个体一天的时空轨迹长什么样

  + fig_stage2_clusters 某用户的簇划分 + 阶段3 判定 —— 六张必交图里唯一没覆盖的
    「阶段2 到底做了什么」，讲义未要求，作为附加图补上
  + fig_heatmap_compare 图5 两种口径（每小时人次 vs 最挤的 10 分钟窗口）上下对照，
    答辩被问「为什么你的热力图和示例不是一个量级」时直接翻这张
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
HEATMAP_TOP = 10            # 热力矩阵取全天人次最高的前 10 个商圈（本项目商圈表一共 5 个）
HEATMAP_LABEL_RATIO = 0.5   # 只在 >= 全表最大值一半的格子里写数字（示例图的画法）
# 3D 轨迹的连线配色：五个人的线要能一眼分开，且不能和「红=出行、绿=驻留」撞色
USER_COLORS = ["#9467bd", "#ff7f0e", "#1f77b4", "#8c564b", "#17becf"]


def _save(fig, path):
    # bbox_inches="tight" 与示例代码同款：既裁掉画布四周的空白，又保证贴到边界外的
    # 图例（图1）不会被切掉一半。
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  图已保存 {path}")


def _hour_of(series):
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S", errors="coerce").dt.hour


def plot_stage3_pie(label_df):
    """图1：驻留/出行占比饼图。

    排版按示例图：扇区不弹出，百分比写在扇区内，具体点数放在右侧图例里，
    这样饼图本体不会被「驻留 13,000 点」这种长标签挤变形。
    """
    stay = int((label_df["label"] == 0).sum())
    travel = int((label_df["label"] == 1).sum())
    fig, ax = plt.subplots(figsize=(6, 6))
    # 带 autopct 时 ax.pie 返回三元组（wedges, labels, percentages）
    wedges, _, autotexts = ax.pie([stay, travel], autopct="%1.1f%%",
                                  colors=[GREEN, RED], startangle=90,
                                  wedgeprops={"edgecolor": "white", "linewidth": 1})
    for t in autotexts:                                # 黑字压在深绿/深红上看不清
        t.set_color("white")
        t.set_fontsize(11)
    ax.legend(wedges, [f"驻留（{stay:,} 点）", f"出行（{travel:,} 点）"],
              loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
    ax.set_title(f"出行 / 驻留标签分布（共 {stay + travel:,} 点）")
    _save(fig, config.FIG_STAGE3_PIE)


def plot_hour_line(label_df):
    """图2：24 小时出行/驻留叠在同一张图上（对齐讲义示例的排版）。

    低频点改为保留、出发地也计出行后，两条线量级接近（本次驻留约为出行的 2 倍），共轴直接画，
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
    # 出行线下方淡红填充：示例图就有这块底，红线本身在 24 个点里不容易读面积
    ax.fill_between(hours, travel.values, color=RED, alpha=.10)
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

    fig.suptitle("24 小时出行 / 驻留点分布")
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
    ax.grid(axis="y", linestyle="--", alpha=.4)      # 示例图柱底有横向虚线网格
    ax.set_axisbelow(True)
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
    """图5：Top10 商圈 × 24 小时热力矩阵，格子里是**该小时的真实命中人次**，按绝对人次上色。

    数值来自 heat_hourly.csv，它在 10 分钟滑动窗口之前统计（见 `heatmap.hourly_counts`）。
    之前这里直接吃 heatmap_result.csv，两种取法都不对：按小时求和会把同一个人重复数 10 次
    （一小时 60 个互相重叠的窗口），取 .max() 又只反映最挤的那一格——太原街 23 时 2,570，
    而示例同一格 32,187，差 12 倍。两张图画的本就不是一个量，不是数据差那么多。
    换成小时人次后太原街 23 时 = 12,502，与示例同为万级，逐时曲线归一化后与示例 r=0.946。

    色标全表共用绝对刻度，不按行归一化：否则棋盘山（全天 79）会画得和太原街（13 万）一样红。
    数字只写深色格（>= 全表最大值一半），示例图就是这个画法；行尾不加「峰值 X」列，
    那列文字会把 xlim 撑出去、在矩阵右边拉出一条再没有内容的白边。

    标题照示例图写「Top10商圈每小时人次热力图」，不带口径后缀；命中半径
    config.HEATMAP_MATCH_THRESHOLD 米（当前 1000m，最近商圈）这条只在色标和这里说明。
    """
    heat_df = heat_df.sort_values("全天合计", ascending=False).head(HEATMAP_TOP)
    names = heat_df["biz_name"]
    M = heat_df.drop(columns=["biz_name", "全天合计"])

    values = M.to_numpy(dtype=float)
    vmax = values.max()
    fig, ax = plt.subplots(figsize=(11.5, 1.45 * len(M) + 1.2))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(24), [str(h) for h in range(24)], fontsize=8)
    ax.set_yticks(range(len(M)), names)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            if v >= vmax * HEATMAP_LABEL_RATIO:
                # 不加千分位：一格只有 50 像素宽，"10,864" 会和隔壁的 "12,502" 挤成
                # 一个十位数。示例图同样是裸数字。
                ax.text(j, i, int(v), ha="center", va="center",
                        fontsize=7.5, color="white")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(len(M) - 0.5, -0.5)
    ax.set_xlabel("小时")
    ax.set_ylabel("商圈")
    ax.set_title("Top10商圈每小时人次热力图")
    fig.colorbar(im, ax=ax, label="小时命中人次", pad=0.02)
    _save(fig, config.FIG_HEATMAP)


def plot_heatmap_compare(heat_df, window_file):
    """附加图：图5 的两种口径上下同框对照（图5 本体不动）。

    上格＝图5 现用口径「每小时命中人次」（heat_hourly.csv，在滑动窗口之前数的）；
    下格＝老师示例代码的取法：从 heatmap_result.csv 按 (商圈, 小时) 取 `.max()`，
    也就是该小时里最挤的那一个 10 分钟窗口。

    两格**共用一条色标**：这张图要说的就是量级差，各自归一化会把两格都画满红，
    对照就废了。代价是下格整体偏淡——那正是它只数了 1/6 小时的直观样子。
    """
    win = read_csv(window_file)
    # read_csv 默认 dtype=str（护住 imsi），这里不转回数字的话，
    # groupby.max() 会按字符串比大小，"998" 会赢过 "1318"。
    win["count"] = pd.to_numeric(win["count"], errors="coerce")
    win["hour"] = pd.to_datetime(win["time"]).dt.hour
    peak = (win.groupby(["biz_name", "hour"])["count"].max().unstack(fill_value=0)
            .reindex(columns=range(24), fill_value=0))

    cols = [str(h) for h in range(24)]
    names = list(heat_df["biz_name"])
    A = heat_df.set_index("biz_name").reindex(names)[cols].to_numpy(dtype=float)
    B = peak.reindex(names).fillna(0).to_numpy(dtype=float)
    vmax = max(A.max(), B.max())
    ratio = A.max() / max(B.max(), 1)
    rs = [pd.Series(a).corr(pd.Series(b)) for a, b in zip(A, B)]
    rmin = min(r for r in rs if pd.notna(r))

    fig, axes = plt.subplots(2, 1, figsize=(13, 1.45 * len(names) * 2 + 3.0),
                             sharex=True, sharey=True)
    for ax, M, sub in zip(axes, (A, B),
                          (f"每小时命中人次（图5 现用口径，全天 {int(A.sum()):,} 人次，"
                           f"单格最高 {int(A.max()):,}）",
                           f"该小时最挤的 10 分钟窗口（示例代码 .max() 口径，全天 {int(B.sum()):,} "
                           f"人次，单格最高 {int(B.max()):,}）")):
        im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        own = M.max()
        for i, row in enumerate(M):
            for j, v in enumerate(row):
                if v >= own * HEATMAP_LABEL_RATIO:
                    ax.text(j, i, int(v), ha="center", va="center", fontsize=7.5,
                            color="white" if v >= vmax * 0.6 else "black")
        ax.set_title(sub, fontsize=10, loc="left")
    fig.colorbar(im, ax=axes, label="人次（两格同一把尺）", pad=0.02)
    axes[1].set_xticks(range(24), [str(h) for h in range(24)], fontsize=8)
    axes[0].set_yticks(range(len(names)), names)
    axes[1].set_xlabel("小时")
    for ax in axes:
        ax.set_ylim(len(names) - 0.5, -0.5)
    fig.suptitle(f"同一批流数据、两种口径：逐时形状一致（r≥{rmin:.2f}），"
                 f"峰值差 {ratio:.1f} 倍")
    _save(fig, config.FIG_HEAT_COMPARE)


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
    ax.set_title(f"Top{len(top_users)}用户3D移动轨迹（经度-纬度-时间）")
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
    """读结果表，生成六张必交图 + 三张附加图；单张图失败只警告，不影响其他图。"""
    log_step(8, "可视化（6 张图 + 3 张附加图）")
    setup_zh()

    label_df = read_csv(config.STAGE3_LABEL_FILE,
                        dtype={"imsi": str, "time": str, "lac_id": str, "cell_id": str})
    label_df["label"] = pd.to_numeric(label_df["label"], errors="coerce")
    label_df["lon"] = pd.to_numeric(label_df["lon"], errors="coerce")
    label_df["lat"] = pd.to_numeric(label_df["lat"], errors="coerce")
    purpose_df = read_csv(config.PURPOSE_RESULT_FILE, dtype={"imsi": str})
    mode_df = read_csv(config.MODE_RESULT_FILE, dtype={"imsi": str})
    heat_df = read_csv(config.HEAT_HOURLY_FILE)
    heat_df["全天合计"] = pd.to_numeric(heat_df["全天合计"], errors="coerce")
    cluster_df = read_csv(config.STAGE2_CLUSTER_FILE, dtype={"imsi": str, "time": str})
    cluster_df["cluster"] = pd.to_numeric(cluster_df["cluster"], errors="coerce")

    jobs = [("出行/驻留饼图", plot_stage3_pie, (label_df,)),
            ("24小时双折线", plot_hour_line, (label_df,)),
            ("出行目的柱状", plot_purpose_bar, (purpose_df,)),
            ("出行方式柱状", plot_mode_bar, (mode_df,)),
            ("商圈热力矩阵", plot_heatmap, (heat_df,)),
            ("Top5 3D 轨迹", plot_3d_traj, (label_df,)),
            ("阶段2簇划分", plot_stage2_clusters, (cluster_df, label_df)),
            ("图5两种口径对照", plot_heatmap_compare, (heat_df, config.HEATMAP_RESULT_FILE))]
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
