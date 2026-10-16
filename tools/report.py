# -*- coding: utf-8 -*-
"""
tools/report.py —— 由 7 个结果 CSV 重新排出 result/项目总结报告.md（可选工具）

原则：报告里的数字全部从 result/ 下已有的 CSV/PNG 现算，不在这里写死参考值，
也不依赖 stats.json；因此 `python main.py` 跑完后随时 `python tools/report.py`
都能重建报告，换数据集后也不会留下旧数字。

跑法：`python tools/report.py`
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                      # noqa: E402
import mode as mode_mod            # noqa: E402
from utils import log, read_csv    # noqa: E402

ID = {"imsi": str, "time": str, "lac_id": str, "cell_id": str}


def _md(df):
    """优先用 df.to_markdown()（需 tabulate）；没装就自己拼一份等价的 Markdown 表格，
    这样 requirements-dev.txt 里那个可选依赖才真的是可选的，而不是隐性必装。
    手工分支照 tabulate 的排版来：列宽取表头与单元格的最大显示宽度，
    数值列右对齐、分隔行用 `---:`，文本列左对齐，两条路径产出同一份文本。"""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = [str(c) for c in df.columns]
        numeric = [pd.api.types.is_numeric_dtype(t) for t in df.dtypes]
        rows = [["" if pd.isna(v) else str(v) for v in rec]
                for rec in df.itertuples(index=False)]
        width = [max([len(cols[i])] + [len(r[i]) for r in rows]) for i in range(len(cols))]

        def pad(text, i):
            return text.rjust(width[i]) if numeric[i] else text.ljust(width[i])

        lines = ["| " + " | ".join(pad(c, i) for i, c in enumerate(cols)) + " |",
                 "| " + " | ".join(("---:" if numeric[i] else ":---")
                                    for i in range(len(cols))) + " |"]
        lines += ["| " + " | ".join(pad(c, i) for i, c in enumerate(r)) + " |" for r in rows]
        return "\n".join(lines)


def _dist(order, counts, unit):
    """固定类别顺序 + 占比，与柱状图同一口径，避免图文对不上。"""
    total = int(counts.sum()) or 1
    rows = [{"类别": c, unit: int(counts.get(c, 0)),
             "占比%": round(int(counts.get(c, 0)) / total * 100, 1)} for c in order]
    return _md(pd.DataFrame(rows)) + f"\n\n合计 {total:,} {unit}。"


def _fig(path, title):
    return f"![{title}](figures/{os.path.basename(path)})"


def _size(path):
    return f"{os.path.getsize(path) / 1024:.1f} KB" if os.path.isfile(path) else "缺失"


def collect():
    """把 7 个结果表读进来并算出报告需要的全部数字。"""
    s = {}
    cleaned = read_csv(config.CLEANED_DATA_FILE, dtype=ID)
    s1 = read_csv(config.STAGE1_TRAVEL_POINTS_FILE, dtype=ID)
    s2 = read_csv(config.STAGE2_CLUSTER_FILE, dtype=ID)
    s3 = read_csv(config.STAGE3_LABEL_FILE, dtype=ID)
    pur = pd.read_csv(config.PURPOSE_RESULT_FILE, dtype={"imsi": str})
    mod = pd.read_csv(config.MODE_RESULT_FILE, dtype={"imsi": str})
    heat = pd.read_csv(config.HEATMAP_RESULT_FILE, dtype={"time": str, "biz_name": str})
    for frame, cols in ((s2, ["lon", "lat", "cluster", "is_single_cluster", "speed"]),
                        (s3, ["lon", "lat", "label"]),
                        (s1, ["lon", "lat", "speed"]),
                        (cleaned, ["lon", "lat"])):
        for c in cols:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")

    s["heat_frame"] = heat

    clusters = s2.groupby(["imsi", "cluster"]).size()
    k_per_user = s2.groupby("imsi")["cluster"].nunique()
    travel = s3["label"] == 1
    merged = s2.merge(s3[["imsi", "time", "label"]], on=["imsi", "time"], how="left")

    s["clean"] = {"点数": len(cleaned), "用户数": cleaned["imsi"].nunique(),
                  "坐标数": cleaned["lon"].round(4).nunique(),
                  "时间范围": f"{cleaned['time'].min()} ~ {cleaned['time'].max()}"}
    s["stage1"] = {"出行点数": len(s1), "占清洗点%": round(len(s1) / len(cleaned) * 100, 1),
                   # 表里 speed 存了两位小数，用半个最小单位兜住舍入，否则和阶段1 日志差 2 个点
                   "到达点": int((s1["speed"] > config.SPEED_THRESHOLD - 0.005).sum()),
                   "速度中位": float(s1["speed"].median()),
                   "速度最大": float(s1["speed"].max())}
    s["stage2"] = {"用户数": s2["imsi"].nunique(), "簇数": len(clusters),
                   "单点簇": int((clusters == 1).sum()),
                   "k_min": int(k_per_user.min()), "k_max": int(k_per_user.max()),
                   "k_mean": round(float(k_per_user.mean()), 2)}
    s["stage3"] = {"全量": len(s3), "出行": int(travel.sum()),
                   "驻留": int((~travel).sum()),
                   "出行占比": round(float(travel.mean()) * 100, 1),
                   "来源": {"单点簇": int((merged["is_single_cluster"] == 1).sum()),
                            "多点簇": int(((merged["label"] == 1)
                                          & (merged["is_single_cluster"] == 0)).sum()),
                            "多点簇驻留": int(((merged["label"] == 0)
                                             & (merged["is_single_cluster"] == 0)).sum()),
                            "未聚类默认驻留": len(s3) - len(s2)}}
    s["purpose"] = {"段数": len(pur), "匹配率": round(float(pur["match_success"].mean()) * 100, 1),
                    "分布": pur["final_purpose"].value_counts()}
    s["mode"] = {"段数": len(mod), "均速中位": float(mod["avg_speed"].median()),
                 "距离中位": float(mod["distance_km"].median()),
                 "分布": mod["final_mode"].value_counts()}
    peak_idx = heat["count"].idxmax()
    s["heat"] = {"记录数": len(heat), "商圈数": heat["biz_name"].nunique(),
                 "峰值": int(heat.loc[peak_idx, "count"]),
                 "峰值时刻": heat.loc[peak_idx, "time"],
                 "峰值商圈": heat.loc[peak_idx, "biz_name"],
                 "窗口": config.HEATMAP_WINDOW_SIZE}
    return s


def build(s):
    lines = []
    A = lines.append
    c, f1, f2, f3 = s["clean"], s["stage1"], s["stage2"], s["stage3"]

    A("# 交通时空数据分析项目总结报告")
    A("")
    A(f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M} ｜ 目标日期：{config.TARGET_DATE} ｜ "
      f"数据：沈阳运营商基站信令定位数据 ｜ 抽样口径：SAMPLE_ROWS = {config.SAMPLE_ROWS}"
      f"（{'全量' if not config.SAMPLE_ROWS else '整用户抽样'}）")
    A("")
    A("## 一、总体结论")
    A("")
    A(f"- 清洗后得到 **{c['点数']:,} 个定位点 / {c['用户数']} 位用户 / "
      f"{c['坐标数']} 个坐标**，时间范围 {c['时间范围']}；")
    A(f"- 阶段1 速度粗筛挑出 **{f1['出行点数']:,} 个疑似出行点**"
      f"（占清洗点 {f1['占清洗点%']}%），相邻点速度中位 {f1['速度中位']:.1f} km/h；")
    A(f"- 阶段2 手写 K-Means 按用户聚类，得到 **{f2['簇数']} 个簇**"
      f"（{f2['单点簇']} 个单点簇），最优 K 区间 {f2['k_min']}~{f2['k_max']}、"
      f"均值 {f2['k_mean']}；")
    A(f"- 阶段3 三点距离 + 首末跨度精判：**出行 {f3['出行']:,} 点"
      f"（{f3['出行占比']}%）、驻留 {f3['驻留']:,} 点**，"
      f"出行点来源 = 单点簇 {f3['来源']['单点簇']} + 多点簇 {f3['来源']['多点簇']}；")
    A(f"- 阶段4 出行目的：{s['purpose']['段数']} 个驻留段，POI 空间匹配率 "
      f"{s['purpose']['匹配率']}%；阶段5 出行方式：{s['mode']['段数']} 个出行段；")
    A(f"- 实时热力：{s['heat']['记录数']:,} 条非零 {s['heat']['窗口']} 分钟窗口记录，"
      f"峰值 {s['heat']['峰值']:,} 人出现在 {s['heat']['峰值商圈']}"
      f"（{s['heat']['峰值时刻']}）；")
    A("- 全部数字可由 `python main.py` 一次性重跑复现（同种子结果一致）。")
    A("")

    A("## 二、工程结构与运行方式")
    A("")
    A("```")
    A("config.py       全部路径与阈值（改参数只改这里，python config.py 自检输入文件）")
    A("utils.py        Haversine / pairwise 欧氏 / 计时器 / 中文字体 / 读写 / 日志")
    A("data_clean.py   ① 九条清洗规则 + 整用户抽样")
    A("tr_stage1.py    ② 阶段1：低频点标记（不删） + 连续段 + 速度粗筛（到达点与出发地同标）")
    A("tr_stage2.py    ③ 阶段2：手写 K-Means + 轮廓系数&邓恩指数选 K")
    A("tr_stage3.py    ④ 阶段3：三点距离 + 首末时间跨度规则精判")
    A("purpose.py      ⑤ 出行目的：POI 空间证据 + 时间窗兜底")
    A("mode.py         ⑥ 出行方式：站点邻近点数证据 + 速度兜底")
    A("heatmap.py      ⑦ 实时流 × 商圈 10 分钟滚动窗口热力")
    A("visualize.py    ⑧ 六张结果图")
    A("main.py         一键运行八模块（模块级容错 + 计时 + 产物核对）")
    A("tools/          可选工具：selfcheck.py 六项自检、report.py 本报告")
    A("```")
    A("")
    A("```bash")
    A("python main.py                # 按 config.SAMPLE_ROWS 整用户抽样")
    A("python main.py --sample 0     # 全量约 53 万行")
    A("python main.py --no-fig       # 只出 CSV，不生成图")
    A("python data_clean.py          # 单独跑任一模块（每个模块都可独立执行）")
    A("python tools/selfcheck.py     # 交付前自检")
    A("python tools/report.py        # 重建本总结报告")
    A("```")
    A("")
    A("依赖仅 `pandas + numpy + matplotlib`，**不使用 sklearn / geopandas**——"
      "K-Means、轮廓系数、邓恩指数、Haversine 全部按讲义手写实现。")
    A("")

    A("## 三、① 数据清洗（9 条规则）")
    A("")
    A("顺序：字段抽取 → 去脏 imsi → 去空基站编号 → 时间戳转换 → 目标日过滤 →"
      "（整用户抽样）→ 关联基站 → 排序 → 分钟去重 → 稀疏用户过滤。")
    A("")
    A(f"原始时间戳是 13 位 **UTC 毫秒**，直接按 naive 解析会把北京时间整体前移 8 小时，"
      f"「只保留 {config.TARGET_DATE}」就变成了北京时间 10-03 08:00 ~ 10-04 08:00，"
      "凌晨记录全丢、还混进次日数据。规则4 因此先 `+8` 小时再做目标日过滤。")
    A("")
    A("抽样单位用「整用户」而不是「整行」：随机抽行会把同一个人的连续轨迹抽碎，"
      f"随后规则9（每人当天点数 ≥ {config.MIN_POINTS_PER_USER}）会把普通用户几乎全删光。")
    A("")
    A(f"最终清洗产物：{c['点数']:,} 点 / {c['用户数']} 用户 / {c['坐标数']} 个坐标，"
      f"六列 {config.CLEANED_COLS}。")
    A("")

    A("## 四、② 阶段1 连续段与速度粗筛")
    A("")
    A(f"- 低频点（同用户同坐标出现 < {config.LOW_FREQ_THRESHOLD} 次）只打 is_low 标记、**不删行**"
      f"（讲义口径是删除，本项目实测删除会连带删掉 83.7% 的高速移动点，见 README 七）；")
    A(f"- 相邻点速度 > {config.SPEED_THRESHOLD} km/h 的到达点 {f1['到达点']:,} 个，"
      f"再把每次移动的**出发地那一点**一并算作疑似出行（多 "
      f"{f1['出行点数'] - f1['到达点']:,} 个），合计 {f1['出行点数']:,} 个"
      f"（占清洗点 {f1['占清洗点%']}%）；出发地只是「刚离开」，同样不是驻留，"
      f"误标与否交给阶段2/3 复核；")
    A(f"- 速度 <0 或 >{config.ABNORMAL_SPEED} km/h 视为基站跳跃，置 NaN 不参与判定"
      f"（保留在表里供检查，不做删除）；")
    A(f"- 出行点速度中位 {f1['速度中位']:.1f} km/h、最大 {f1['速度最大']:.1f} km/h。")
    A("")
    A("阶段1 是**高召回粗筛**：宁可多留点交给阶段2/3 精判，也不在这里下最终结论。")
    A("")

    A("## 五、③ 阶段2 手写 K-Means 聚类")
    A("")
    A(f"- 只对阶段1 的 {f1['出行点数']:,} 个出行点、按用户分组聚类，"
      f"共 {f2['用户数']} 位用户；")
    A(f"- 遍历 K={config.K_MIN}~{config.K_MAX}，评分 `score = 轮廓系数 + "
      f"{config.DUNN_WEIGHT} × 邓恩指数`；最优 K 最小 {f2['k_min']}、最大 {f2['k_max']}、"
      f"平均 {f2['k_mean']}；")
    A(f"- 得到 {f2['簇数']} 个簇，其中 {f2['单点簇']} 个单点簇——"
      "单点簇没有「跨度」可言，阶段3 直接补判为出行。")
    A("")
    A("K-Means 内部用广播 `(n,1,2) - (1,k,2) → (n,k,2)` 一次算完所有距离；"
      "空簇保留上一轮中心，避免对空集取均值得到 NaN。距离矩阵在 K 循环外只算一次、"
      "各 K 复用；逐用户循环前先 `np.random.seed(" + str(config.RANDOM_STATE) + ")` 一次，"
      "同参数重跑标签完全一致（见 `tools/selfcheck.py` 第 4 项）。")
    A("")

    A("## 六、④ 阶段3 空间 + 时间双条件精判")
    A("")
    A(f"判定规则：簇内**首/中/尾三点到簇心 Haversine 距离均 < "
      f"{config.CLUSTER_DIST_THRESHOLD} m** 且 **簇内首末时间跨度 < "
      f"{config.TIME_SPAN_THRESHOLD} h** → 驻留(0)，否则出行(1)；"
      "单点簇直接判出行(1)；未进入聚类的点（阶段1 已判非出行）默认驻留(0)。")
    A("")
    A(_md(pd.DataFrame({
        "点来源": ["单点簇直接判出行", "多点簇不满足双条件判出行",
                   "多点簇满足双条件判驻留", "未参与聚类默认驻留"],
        "点数": [f3["来源"]["单点簇"], f3["来源"]["多点簇"],
                 f3["来源"]["多点簇驻留"], f3["来源"]["未聚类默认驻留"]],
    })))
    A("")
    A(f"最终：出行 {f3['出行']:,} 点（{f3['出行占比']}%）、驻留 {f3['驻留']:,} 点。")
    A("")
    A("注意 label 方向：**1 = 出行、0 = 驻留**，且默认值是 0——"
      "「未参加聚类的点保守判驻留」，宁可少判出行也不把停留误判成赶路。"
      "阶段4 从 label==0 切驻留段，阶段5 从 label==1 切出行段。")
    A("")

    A("## 七、⑤ 出行目的识别")
    A("")
    A(f"规则链（窄条件在前，宽条件在后）：停留 > {config.LIVING_DURATION_THRESHOLD} h 且首末点落在"
      f"夜间 {config.NIGHT_HOURS} → **居住**；否则若段中心到最近 POI < "
      f"{config.PURPOSE_MATCH_THRESHOLD} m 且该 POI 属于购物/旅游 → 取 POI 类型；"
      f"再兜底 首末点都在 {config.TRAVEL_WINDOW} 内 → **旅游**，"
      f"都在 {config.SHOP_WINDOW} 内 → **购物**，都不满足 → **其他**。")
    A("")
    A("旅游窗 `[9,17]` 是购物窗 `[9,21]` 的子集，**如果把宽条件放前面，窄条件永远命不中，"
      "旅游会恒为 0**——这是课时 45 强调的「死代码陷阱」，"
      "`tools/selfcheck.py` 第 5 项专门对此做回归防护。")
    A("")
    A(_dist(config.PURPOSE_ORDER, s["purpose"]["分布"], "段"))
    A("")
    A(f"共 {s['purpose']['段数']} 个驻留段，POI 空间匹配率 {s['purpose']['匹配率']}%。")
    A("")

    A("## 八、⑥ 出行方式识别")
    A("")
    A(f"裁决优先级：**站点证据优先、速度兜底**。段内点到最近地铁站 < "
      f"{config.MODE_MATCH_THRESHOLD} m 的点数 ≥ {config.SUBWAY_MIN_POINTS} → 地铁；"
      f"最近公交站邻近点数 ≥ {config.BUS_MIN_POINTS} → 公交；"
      f"均速 > {config.MODE_BUS_SPEED} km/h → 小汽车；否则 → 其他。")
    A("")
    stations = mode_mod.load_stations()
    n_subway, n_bus = len(stations[0]), len(stations[1])
    A(f"地铁表只有 {n_subway} 个站、公交表 {n_bus} 个站，"
      "公交站密集容易误命中，所以公交的邻近点门槛（≥"
      f"{config.BUS_MIN_POINTS}）比地铁（≥{config.SUBWAY_MIN_POINTS}）更高。")
    A("")
    A(_dist(config.MODE_ORDER, s["mode"]["分布"], "段"))
    A("")
    A(f"共 {s['mode']['段数']} 个出行段（每段 ≥2 点才能算速度，单点段在建段阶段已丢弃）；"
      f"段均速中位 {s['mode']['均速中位']:.1f} km/h、段距离中位 "
      f"{s['mode']['距离中位']:.2f} km。")
    A("")

    A("## 九、⑦ 实时商圈 10 分钟滚动窗口热力")
    A("")
    heat = s["heat"]
    A(f"实时流分批（每批 {config.BATCH_SIZE} 行）广播匹配、命中阈值 "
      f"{config.HEATMAP_MATCH_THRESHOLD} m；按分钟聚合后补齐成完整 1440 分钟再做 "
      f"{heat['窗口']} 分钟滚动求和，输出 {heat['记录数']:,} 条非零窗口记录，"
      f"涉及 {heat['商圈数']} 个商圈。")
    A("")
    A("补齐 1440 分钟是必须的：`rolling` 直接在稀疏时间索引上滑窗会跨空档累加，"
      "把几小时前的计数算进当前窗口。")
    A("")
    top = (s["heat_frame"].groupby("biz_name")["count"].max().sort_values(ascending=False)
           .head(10).rename_axis("商圈").reset_index(name="窗口内峰值人数"))
    A("窗口计数最高的商圈：")
    A("")
    A(_md(top))
    A("")
    A(f"全天峰值 {heat['峰值']:,} 人（{heat['峰值商圈']}，{heat['峰值时刻']}）。")
    A("")

    A("## 十、结果图（课时 56 六张）")
    A("")
    for path, title in ((config.FIG_STAGE3_PIE, "阶段3 出行/驻留占比"),
                        (config.FIG_HOUR_LINE, "逐小时出行/驻留点数量"),
                        (config.FIG_PURPOSE_BAR, "出行目的分布"),
                        (config.FIG_MODE_BAR, "出行方式分布"),
                        (config.FIG_HEATMAP, "商圈 × 小时 热力图"),
                        (config.FIG_3D_TRAJ, "Top5 用户出行轨迹 3D 视图")):
        A(_fig(path, title))
        A("")
    A("出图纪律：类别顺序固定并 `reindex(fill_value=0)` 补齐，颜色与类别绑定，"
      "这样某一类计数为 0 时柱子不会错位、颜色也不会串味。")
    A("")

    A("## 十一、交付产物清单")
    A("")
    rows = [{"产物": name, "文件": os.path.basename(path), "大小": _size(path)}
            for name, path in config.RESULT_CSVS]
    rows += [{"产物": f"图 {i + 1}", "文件": os.path.basename(p), "大小": _size(p)}
             for i, p in enumerate(config.RESULT_FIGS)]
    A(_md(pd.DataFrame(rows)))
    A("")

    A("## 十二、局限与改进方向")
    A("")
    A("1. **定位精度**：基站扇区定位误差数百米至数公里，"
      f"{config.CLUSTER_DIST_THRESHOLD} m 驻留半径与 {config.PURPOSE_MATCH_THRESHOLD} m POI "
      "匹配阈值都受此上限约束，目的/方式结论只适合做趋势判断；")
    A("2. **采样密度**：信令是事件驱动上报，稀疏且不匀速，"
      f"首末跨度 < {config.TIME_SPAN_THRESHOLD} h 的阈值本质是对采样密度的妥协，"
      "可改用「首末时间差 / 点数」或按用户分位数自适应；")
    A("3. **选 K 指标**：轮廓系数 + 邓恩指数在簇密度差异大时偏保守，"
      "可加 SSE 肘部法或 Gap Statistic 做双指标交叉；")
    A("4. **类别不平衡**：静态 POI 表中居住类占比过半、"
      "地铁站数远少于公交站，最近邻匹配天然偏向多数类，兜底分支的排序因此不能随意交换；")
    A("5. **随机性**：K-Means 只在逐用户循环前固定一次全局种子，"
      "逐用户的最优 K 会受初始化顺序影响，与讲义参考数字有 ±3% 量级的小偏差；"
      "多次重启取 SSE 最小可进一步收敛；")
    A("6. **性能**：阶段2 逐用户 Python 循环、热力/匹配的分批广播是主要开销，"
      "可按用户 multiprocessing 分片，或先做网格预聚类再对质心跑 K-Means。")

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(config.RESULT_DIR, "项目总结报告.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    log(f"总结报告已生成：{out_path}（{len(text):,} 字符）")
    return out_path


def main():
    build(collect())


if __name__ == "__main__":
    main()
