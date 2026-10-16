# -*- coding: utf-8 -*-
"""
config.py —— 全局配置：目录路径 + 输入文件 + 7 个结果 CSV + 6 张图路径 + 全部业务阈值。

约定（课时 04）：路径与阈值只在本文件出现一次，业务模块 `import config` 读取，
代码里不写死路径、不出现魔法数字。

`python config.py` 会打印全部配置并检查 6 个输入数据文件是否就位。
"""
import os

# ----------------------------------------------------------------------
# 目录结构：项目根 = 本文件所在目录（traffic_project/）
#   result/          7 个结果 CSV
#   result/figures/  6 张 PNG
# 数据目录不在项目内，由 _find_data_dir() 自动定位（也可用环境变量指定）
# ----------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_data_dir():
    """按 环境变量 → 项目同级/上级 → 两层目录扫描 的顺序定位数据目录。

    只认「目录里存在原始数据.csv」，避免误命中同名的其它 csv。
    """
    env = os.environ.get("TRAFFIC_DATA_DIR")
    if env and os.path.isfile(os.path.join(env, "原始数据.csv")):
        return env

    parent = os.path.dirname(PROJ_DIR)          # 仓库根（TST-Analytics/）
    candidates = [
        os.path.join(PROJ_DIR, "data"),           # 讲义兜底：6 个 CSV 直接放包内
        os.path.join(parent, "data", "raw"),      # 本仓库：TST-Analytics/data/raw
        os.path.join(parent, "交通时空数据分析数据"),  # 本仓库：原始发放目录
        os.path.join(parent, "data"),
        os.path.join(parent, "相关数据"),           # 讲义里的目录名
    ]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "原始数据.csv")):
            return cand

    # 前面都没命中再全盘扫两级，兜住「包被拷到别处」的情况
    for root, dirs, _files in os.walk(parent, topdown=True):
        depth = os.path.relpath(root, parent).count(os.sep) if root != parent else 0
        dirs[:] = [d for d in dirs if not d.startswith((".", "课件")) and depth < 2]
        if os.path.isfile(os.path.join(root, "原始数据.csv")):
            return root
    return candidates[0]


DATA_DIR = _find_data_dir()
RESULT_DIR = os.path.join(PROJ_DIR, "result")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

for _d in (RESULT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------
# 六大数据文件（课时 06；列序按真实文件，不按讲义早期假设）
# ----------------------------------------------------------------------
RAW_DATA_FILE = os.path.join(DATA_DIR, "原始数据.csv")           # 532828 行 × 10 列，无表头，UTF-8
STATION_DATA_FILE = os.path.join(DATA_DIR, "基站经纬度数据.csv")  # 49952 行：lon,lat,laci
BIZ_DATA_FILE = os.path.join(DATA_DIR, "商业区数据.csv")          # 104 行：lon,lat,biz_name
PURPOSE_POI_FILE = os.path.join(DATA_DIR, "出行目的静态数据.csv")  # 1273 行：lon,lat,poi_type,poi_name，GBK
MODE_STATION_FILE = os.path.join(DATA_DIR, "出行方式静态数据.csv")  # 968 行：lon,lat,mode_type,station,line，GBK
STREAM_DATA_FILE = os.path.join(DATA_DIR, "实时流数据.csv")       # 256212 行：time,imsi,lon,lat

# ----------------------------------------------------------------------
# 阶段0：数据清洗（课时 23~30 九条规则）
# ----------------------------------------------------------------------
RAW_COLUMNS = ["time", "imsi", "lac_id", "cell_id"]  # 规则1：原始 10 列只取这 4 列
TARGET_DATE = "20181003"          # 规则5：仅保留目标日
RAW_TIMEZONE = "Asia/Shanghai"  # 规则4：原始是 UTC 毫秒戳，按 IANA 时区名转北京时间（纪要 day2：不写死 +8）
MIN_POINTS_PER_USER = 20          # 规则9：当天点数 < 20 的稀疏用户剔除
# 规则6：补完经纬度后过滤异常坐标。区间出自《项目指导会议纪要》day1「基站经纬度补全与
# 坐标过滤」：经度 122.5~124.5、纬度 41.0~42.5。实测越界点 1,333 行（0.26%），最远
# lat 43.15 已在沈阳以北约 200km，属基站定位漂移，留着会把阶段2 的簇心拖偏。
SIGNAL_LON_RANGE = (122.5, 124.5)
SIGNAL_LAT_RANGE = (41.0, 42.5)
SAMPLE_ROWS = 20000               # 课时 28：整用户抽样目标行数；0 = 不抽样
RANDOM_STATE = 42                 # 抽样与聚类的统一随机种子

# ----------------------------------------------------------------------
# 阶段1：低频点 + 切段 + 相邻点速度（课时 31~34）
# ----------------------------------------------------------------------
LOW_FREQ_THRESHOLD = 4            # 课时 32：同用户同坐标 < 4 次视为低频，只标记 is_low 不删点
SPEED_THRESHOLD = 8.0             # 课时 33：相邻点速度 > 8 km/h → 疑似出行点
ABNORMAL_SPEED = 200              # 速度 <0 或 >200 km/h 视为基站跳跃异常，置 NaN

# ----------------------------------------------------------------------
# 阶段2：手写 K-Means + 轮廓系数/邓恩指数选 K（课时 35~38）
# ----------------------------------------------------------------------
K_MIN = 2
K_MAX = 10                        # 遍历 K = 2~10
KM_MAX_ITER = 100                 # 课时 36：迭代上限
KM_TOL = 0.1                      # 中心位移收敛阈值，单位米（聚类输入已由 utils.project_to_meters 投影）
DUNN_WEIGHT = 0.1                 # 课时 37：score = 轮廓系数 + 0.1 × 邓恩指数

# ----------------------------------------------------------------------
# 阶段3：三点距离 + 首末时间跨度精判（课时 39~42）
# ----------------------------------------------------------------------
CLUSTER_DIST_THRESHOLD = 1000     # 首/中/尾三点到簇心 Haversine < 1000m
TIME_SPAN_THRESHOLD = 2           # 簇内首末时间跨度 < 2h → 整簇判驻留

# ----------------------------------------------------------------------
# 阶段4：出行目的（课时 43~46）
# ----------------------------------------------------------------------
LIVING_DURATION_THRESHOLD = 4     # 居住：停留 > 4h 且端点在夜间
PURPOSE_MATCH_THRESHOLD = 1000    # 段中心到最近 POI < 1000m 才算有效空间证据
SHOP_WINDOW = (9, 21)             # 购物时间兜底窗
TRAVEL_WINDOW = (9, 17)           # 旅游时间兜底窗（比购物窄，必须先判）
NIGHT_HOURS = [(0, 6), (20, 23)]  # 夜间时段

# ----------------------------------------------------------------------
# 阶段5：出行方式（课时 47~50）
# ----------------------------------------------------------------------
MODE_MATCH_THRESHOLD = 500        # 段内点到最近站 < 500m 记一个邻近点
SUBWAY_MIN_POINTS = 2             # 地铁站稀疏、证据强：≥2 个邻近点认地铁
BUS_MIN_POINTS = 3                # 公交站密集：门槛提高到 ≥3 个邻近点
MODE_OTHER_SPEED = 12             # <12 km/h 偏慢（步行/骑行）
MODE_BUS_SPEED = 20               # >20 km/h 且无站点证据 → 小汽车

# ----------------------------------------------------------------------
# 实时热力（课时 51~54）
# ----------------------------------------------------------------------
HEATMAP_WINDOW_SIZE = 10          # 10 分钟滑动窗口
HEATMAP_MATCH_THRESHOLD = 1000    # 流点到最近商圈 < 1000m 才算命中
BATCH_SIZE = 10000                # 分批广播，控制内存峰值
# 剔静态表里的脏坐标：商业区数据.csv 第 23 行沈阳北站 lon=23.454567，少了前导的 1。
# 注意上界不能照搬信令分析用的沈阳范围(41.1~41.9)——目的静态表里有 16 行、方式静态表里
# 有 15 行 lat≈42.7 的点，那是康平县的**真实** POI（康平属沈阳市），按窄区间筛会误删。
POI_LNG_RANGE = (122.0, 124.0)
POI_LAT_RANGE = (41.0, 43.5)

# ----------------------------------------------------------------------
# 输出：7 个结果 CSV（课时 57 逐个核对存在性）
# ----------------------------------------------------------------------
CLEANED_DATA_FILE = os.path.join(RESULT_DIR, "cleaned_data.csv")
STAGE1_TRAVEL_POINTS_FILE = os.path.join(RESULT_DIR, "stage1_travel_points.csv")
STAGE2_CLUSTER_FILE = os.path.join(RESULT_DIR, "stage2_cluster.csv")
STAGE3_LABEL_FILE = os.path.join(RESULT_DIR, "stage3_label.csv")
PURPOSE_RESULT_FILE = os.path.join(RESULT_DIR, "purpose_result.csv")
MODE_RESULT_FILE = os.path.join(RESULT_DIR, "mode_result.csv")
HEATMAP_RESULT_FILE = os.path.join(RESULT_DIR, "heatmap_result.csv")
# 图5 用的是这张而不是上面那张：heatmap_result.csv 存的是 60 个互相重叠的 10 分钟窗口，
# 按小时相加会把同一个人重复数 10 次。小时人次必须在 rolling 之前统计，见 heatmap.hourly_counts。
# 它是图5 的中间表，不在课时 57 要核对的 7 个交付 CSV 之列。
HEAT_HOURLY_FILE = os.path.join(RESULT_DIR, "heat_hourly.csv")

RESULT_CSVS = [
    ("清洗数据", CLEANED_DATA_FILE),
    ("阶段1出行点", STAGE1_TRAVEL_POINTS_FILE),
    ("阶段2聚类结果", STAGE2_CLUSTER_FILE),
    ("阶段3标签", STAGE3_LABEL_FILE),
    ("出行目的", PURPOSE_RESULT_FILE),
    ("出行方式", MODE_RESULT_FILE),
    ("实时热力", HEATMAP_RESULT_FILE),
]

# ----------------------------------------------------------------------
# 输出：6 张 PNG（课时 56）
# ----------------------------------------------------------------------
FIG_STAGE3_PIE = os.path.join(FIG_DIR, "fig_stage3_pie.png")
FIG_HOUR_LINE = os.path.join(FIG_DIR, "fig_hour_line.png")
FIG_PURPOSE_BAR = os.path.join(FIG_DIR, "fig_purpose_bar.png")
FIG_MODE_BAR = os.path.join(FIG_DIR, "fig_mode_bar.png")
FIG_HEATMAP = os.path.join(FIG_DIR, "fig_heatmap.png")
FIG_3D_TRAJ = os.path.join(FIG_DIR, "fig_3d_traj.png")

RESULT_FIGS = [FIG_STAGE3_PIE, FIG_HOUR_LINE, FIG_PURPOSE_BAR,
               FIG_MODE_BAR, FIG_HEATMAP, FIG_3D_TRAJ]

# 附加图：讲义只要求上面 6 张，这三张是额外补的交付物，不计入 6 张核对
FIG_CLUSTER = os.path.join(FIG_DIR, "fig_stage2_clusters.png")
FIG_HEAT_COMPARE = os.path.join(FIG_DIR, "fig_heatmap_compare.png")
FIG_DASHBOARD = os.path.join(FIG_DIR, "fig_dashboard.png")

# 早晚高峰底纹（课时 33 的常识区间，只用于图2 标注，不参与任何判定）
PEAK_HOURS = [(7, 9, "早高峰"), (17, 19, "晚高峰")]

# 清洗结果列（课时 29：严格固定 6 列，下游 stage1 依赖）
CLEANED_COLS = ["imsi", "time", "lac_id", "cell_id", "lon", "lat"]

# 课时 56 固定类别顺序：换数据也不会让柱子错位或串色
PURPOSE_ORDER = ["居住", "购物", "旅游", "其他"]
MODE_ORDER = ["地铁", "公交", "小汽车", "其他"]


def check_data_files():
    """返回 (存在的输入文件列表, 缺失的输入文件列表)。"""
    names = [RAW_DATA_FILE, STATION_DATA_FILE, BIZ_DATA_FILE,
             PURPOSE_POI_FILE, MODE_STATION_FILE, STREAM_DATA_FILE]
    have = [n for n in names if os.path.isfile(n)]
    missing = [n for n in names if n not in have]
    return have, missing


if __name__ == "__main__":
    print(f"项目根 : {PROJ_DIR}")
    print(f"数据目录: {DATA_DIR}"
          + ("（来自环境变量 TRAFFIC_DATA_DIR）" if os.environ.get("TRAFFIC_DATA_DIR") else "（自动定位）"))
    print(f"结果目录: {RESULT_DIR}")
    print(f"图片目录: {FIG_DIR}")
    print("-" * 60)
    print(f"目标日期 TARGET_DATE = {TARGET_DATE}")
    print(f"抽样行数 SAMPLE_ROWS = {SAMPLE_ROWS}（0 表示全量）")
    print(f"清洗     低频点<{LOW_FREQ_THRESHOLD}次 ｜ 出行速度>{SPEED_THRESHOLD}km/h ｜ 异常速度>{ABNORMAL_SPEED}km/h ｜ "
          f"坐标限经度{SIGNAL_LON_RANGE[0]}~{SIGNAL_LON_RANGE[1]}、纬度{SIGNAL_LAT_RANGE[0]}~{SIGNAL_LAT_RANGE[1]}")
    print(f"聚类     K={K_MIN}~{K_MAX} ｜ max_iter={KM_MAX_ITER} tol={KM_TOL} ｜ "
          f"score=轮廓+{DUNN_WEIGHT}×邓恩 ｜ seed={RANDOM_STATE}")
    print(f"精判     三点距簇心<{CLUSTER_DIST_THRESHOLD}m 且 首末跨度<{TIME_SPAN_THRESHOLD}h → 驻留")
    print(f"目的     居住>{LIVING_DURATION_THRESHOLD}h且端点夜间 ｜ POI<{PURPOSE_MATCH_THRESHOLD}m ｜ "
          f"兜底 旅游{TRAVEL_WINDOW}→购物{SHOP_WINDOW}")
    print(f"方式     邻近站<{MODE_MATCH_THRESHOLD}m ｜ 地铁≥{SUBWAY_MIN_POINTS}点 ｜ "
          f"公交≥{BUS_MIN_POINTS}点 ｜ 均速>{MODE_BUS_SPEED}→小汽车")
    print(f"热力     窗口{HEATMAP_WINDOW_SIZE}分钟 ｜ 商圈<{HEATMAP_MATCH_THRESHOLD}m ｜ 分批{BATCH_SIZE}")
    print("-" * 60)
    have, missing = check_data_files()
    for n in have:
        print(f"  [OK] {os.path.basename(n)}")
    for n in missing:
        print(f"  [缺失] {os.path.basename(n)}")
    print("-" * 60)
    print(f"输入文件 {len(have)}/6 就位" if not missing else f"!! 缺少 {len(missing)} 个输入文件")
    print(f"结果 CSV 交付 {len(RESULT_CSVS)} 个，PNG 交付 {len(RESULT_FIGS)} 张")
