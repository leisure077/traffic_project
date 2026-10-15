# 出行实训 · 交通时空数据分析（traffic_project）

从沈阳运营商**基站信令定位数据**里识别每个人的**出行 / 驻留**行为，再判断**出行目的**、
**出行方式**，并输出**商圈实时热力**，最后画出 6 张结果图。

一条命令跑完 8 个模块：

```bash
python main.py
```

---

## 一、工程结构

```
traffic_project/
├── config.py        # 所有路径和阈值集中在这里（改参数只改这里）
├── utils.py         # 通用工具：Haversine距离、时间、读写、日志、计时器
├── data_clean.py    # ① 数据清洗（9条规则 + 整用户抽样）
├── tr_stage1.py     # ② 阶段1：速度粗筛
├── tr_stage2.py     # ③ 阶段2：手写K-Means
├── tr_stage3.py     # ④ 阶段3：规则精判
├── purpose.py       # ⑤ 出行目的
├── mode.py          # ⑥ 出行方式
├── heatmap.py       # ⑦ 实时热力
├── visualize.py     # ⑧ 可视化6张图
├── main.py          # 一键运行全部8个模块
├── test_utils.py    # 课时58：unittest 19 项（haversine / 手写kmeans / 毫秒+8时区）
├── requirements.txt
├── README.md
├── tools/           # 可选工具，不在 8 模块链里
│   ├── selfcheck.py #    交付前 6 项自检
│   └── report.py    #    由 7 个结果 CSV 重建总结报告
└── result/
    ├── cleaned_data.csv / stage1_travel_points.csv / stage2_cluster.csv
    ├── stage3_label.csv / purpose_result.csv / mode_result.csv / heatmap_result.csv
    └── figures/     # 6 张 PNG
```

数据目录（`原始数据.csv` 等 6 个输入文件）**不放在项目内**：`config.py` 的
`_find_data_dir()` 按下面的顺序找，第一个含 `原始数据.csv` 的目录即命中：

1. 环境变量 `TRAFFIC_DATA_DIR`；
2. `traffic_project/data/`（讲义的兜底：把 6 个 CSV 直接放包内，包就能脱离仓库跑）；
3. 仓库根的 `data/raw/`（本仓库实际命中这里）；
4. 仓库根的 `交通时空数据分析数据/`、`相关数据/`，以及 `../data/`；
5. 都没有再从仓库根往下扫两层。

全都没命中时 `main.py` 会**直接停住**并打印上面三条出路，而不是带着缺失的输入去跑
八个模块刷八屏 traceback——「没数据」和「某个模块算错」是两类问题，后者才走容错继续。

## 二、环境

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
```

只依赖 `pandas + numpy + matplotlib`（讲义口径的三个库，写在 `requirements.txt`）。
`tools/report.py` 生成 Markdown 表格另需 `tabulate`，已单独放 `requirements-dev.txt`，
不装不影响主流程。
**不使用 sklearn / geopandas**：K-Means、轮廓系数、邓恩指数、Haversine 全部手写。

## 三、运行

```bash
python config.py                  # 打印全部配置 + 检查 6 个输入文件是否就位
python main.py                    # 按 config.SAMPLE_ROWS（默认 20000 行）整用户抽样
python main.py --sample 0         # 全量约 53 万行
python main.py --no-fig           # 只出 7 个 CSV，不生成图
python data_clean.py              # 任一模块都能单独跑（需先跑过它依赖的上游）
python tools/selfcheck.py         # 6 项交付前自检
python -m unittest test_utils -v  # 课时58 单元测试（19 项）
python tools/report.py            # 重建 result/项目总结报告.md
```

`main.py` 是**模块级容错**的：某个模块抛异常只打印 ERROR 并继续跑后面的模块，
教学场景下尽量把已能产出的结果和图都留下，最后统一核对 7 CSV + 6 PNG。

## 四、算法链

```
原始信令 532,828 行（10 列无表头）
   │  ① 清洗：去脏 imsi → 去空编号 → 毫秒UTC+8时区 → 目标日 → 整用户抽样
   │         → 关联基站补经纬度 → 排序 → 分钟去重 → 每人≥20点
   ▼
cleaned_data.csv                        imsi,time,lac_id,cell_id,lon,lat
   │  ② 阶段1：删低频点（同用户同坐标<4次）→ 切连续段 → 相邻点速度>8km/h
   ▼
stage1_travel_points.csv                + speed（高召回粗筛，宁可多留）
   │  ③ 阶段2：经纬度先投影成米（否则欧氏距离把地图横向压扁 25%）→ 逐用户手写 K-Means，
   │           K=2~10 用「轮廓系数 + 0.1×邓恩指数」选优
   ▼
stage2_cluster.csv                      + cluster, is_single_cluster
   │  ④ 阶段3：簇心到首/中/尾三点 <1000m 且 首末跨度 <2h → 驻留，否则出行；
   │           单点簇直接判出行；未参与聚类的点默认驻留（label 1=出行 0=驻留）
   ▼
stage3_label.csv                        全量点 label
   ├─ ⑤ label==0 切驻留段 → POI 空间证据 + 时间窗兜底  → purpose_result.csv
   ├─ ⑥ label==1 切出行段 → 站点邻近点数证据 + 速度兜底 → mode_result.csv
   └─ ⑦ 实时流（自带经纬度）→ 最近商圈匹配 → 分钟×商圈计数
         → 补齐 1440 分钟 → rolling(10) 求和             → heatmap_result.csv
   ⑧ 六张图                                              → result/figures/
```

三条设计线贯穿全程：

- **漏斗式**：阶段1 求召回、阶段2 看结构、阶段3 求准确，每一步都在纠正上一步的偏差；
- **证据优先**：目的与方式都先找空间证据（POI / 站点），证据不足才用时间窗或速度兜底，
  且**窄条件必须排在宽条件前面**（旅游窗 `[9,17]` 是购物窗 `[9,21]` 的子集，
  写反了旅游会恒为 0）；
- **口径统一**：类别顺序、配色在 `config.PURPOSE_ORDER / MODE_ORDER` 里固定，
  报告表格和柱状图用同一份顺序，不会出现图文对不上。

## 五、输出说明

| 文件 | 内容 |
|---|---|
| `cleaned_data.csv` | 清洗后的六列定位点（分析主表） |
| `stage1_travel_points.csv` | 疑似出行点及相邻点速度 |
| `stage2_cluster.csv` | 出行点的簇编号 + 是否单点簇 |
| `stage3_label.csv` | 全量点的出行(1)/驻留(0)标签 |
| `purpose_result.csv` | 驻留段 × 出行目的（含最近 POI 与距离） |
| `mode_result.csv` | 出行段 × 出行方式（含站点证据点数、均速、距离） |
| `heatmap_result.csv` | 每分钟 × 每商圈的近 10 分钟滚动人流 |
| `figures/fig_stage3_pie.png` | 出行/驻留占比 |
| `figures/fig_hour_line.png` | 逐小时出行/驻留双轴曲线 |
| `figures/fig_purpose_bar.png` | 出行目的分布（数量 + 占比） |
| `figures/fig_mode_bar.png` | 出行方式分布（数量 + 占比） |
| `figures/fig_heatmap.png` | 命中商圈 × 24 小时热力矩阵（本次 4 个，见下方口径说明） |
| `figures/fig_3d_traj.png` | Top5 用户出行轨迹 3D 视图（z 轴 = 一天中的时刻） |

结果 CSV 一律 `utf-8-sig` 输出，Excel 双击打开中文不乱码。
`sample=20000` 量级下一次全流程 **4.5~5.9 秒**（本机 2026-10-15 三次实测，Python 3.12 +
numpy 2.2.6 / pandas 2.2.3 / matplotlib 3.10.3），瓶颈在①清洗约 2.6 秒（时间戳转换与
基站关联），其余七步合计约 2 秒。

**静态表坐标的两条口径**（答辩容易被追问，写在这里）：

- 商圈表 `商业区数据.csv` 第 23 行沈阳北站 `lon=23.454567`，经度少了前导的 1。
  读表时按 `config.POI_LNG_RANGE / POI_LAT_RANGE` 剔除并打 WARN 留痕；本次它恰好
  匹配不到任何流点，所以剔与不剔结果相同，但换数据就可能把某个点整条拽飞。
- 5 个商圈里 **沈阳世博园全天 0 命中**（1000m 内没有流点），`groupby` 会把它静默丢掉，
  所以 `match_business()` 专门补一行 WARN 点名，热力图只有 4 行是这件事的结果，
  不是漏算。
- 反过来，`POI_LAT_RANGE` 上界取 **43.5** 而不是信令分析用的 41.9：目的静态表有 16 行、
  方式静态表有 15 行 `lat≈42.7` 的点，那是**康平县**（属沈阳市）的真实 POI，
  按窄区间筛会误删，属于"筛脏数据筛掉好数据"的经典错误。

## 六、自检

`python tools/selfcheck.py` 打印 6 项 PASS/FAIL：

1. 手写 K-Means 能把三团人造数据分对（准确率 > 99%）；
2. 向量化 Haversine 与逐点循环数值一致（相对差 <1e-6，实测差 2.33e-10），且至少快 5 倍
   （各跑 3 轮取最快再算比值——这两段是 11ms/125ms 量级，单次计时受 Windows 时钟粒度、
   调度和冷启动影响很大：同一份代码在仓库目录实测 10.7~11.3×，在刚拷出来的新目录实测
   8.2~12.9×。阈值原先卡在 10× 正好落在噪声带里，会一次 PASS 一次 FAIL，自检结论就不可复现）；
3. 阶段3 不变量：标签仅取 0/1、覆盖清洗全量点、整簇同标、
   出行点数 = 单点簇点数 + 多点簇判出行点数（精确相等），并抽 60 个多点簇用纯 Python
   重算三点判据比对标签；
4. 阶段2 可复现性：同种子两次 `choose_best_k` 标签与最优 K 逐位相同；
5. 目的/方式口径：旅游 > 0（死代码回归防护）、四类齐全、站点证据门槛不被绕过；
6. 交付产物：7 个 CSV + 6 张 PNG 齐套。
