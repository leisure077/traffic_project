# -*- coding: utf-8 -*-
"""
main.py —— 一键集成八大模块 + 结果核对（课时 57）

八个模块各自「读上一步 CSV → 算 → 写下一步 CSV」，这里按顺序串起来，并解决四件事：
  1. 顺序依赖：清洗 → 阶段1 → 阶段2 → 阶段3 → 目的 → 方式 → 热力 → 可视化；
  2. 单模块容错：某个模块报错时打印完整 traceback 但继续跑后面的模块，
     便于一次看清「哪些能跑通、哪些坏了」——教学场景要的是尽量跑完，
     生产系统（如银行交易）通常相反，关键步骤失败即停（fail fast）；
  3. 统一计时：with Timer(name) 自动输出每个模块的耗时；
  4. 跑完自动核对交付物：7 个结果 CSV + 6 张 PNG 是否齐全。

用法：
    python main.py                  # 按 config.SAMPLE_ROWS（默认 2 万行整用户抽样）跑
    python main.py --sample 0       # 全量 53 万行
    python main.py --sample 5000    # 更小的联调样本
    python main.py --no-fig         # 只出 7 个 CSV，不画图
"""
import argparse
import os
import traceback

import config
from data_clean import clean_data
from heatmap import heatmap_analysis
from mode import mode_analysis
from purpose import purpose_analysis
from tr_stage1 import stage1_analysis
from tr_stage2 import stage2_analysis
from tr_stage3 import stage3_analysis
from utils import Timer, log, log_step
from visualize import visualize_all

MODULES = [
    ("数据清洗", clean_data),
    ("阶段1速度粗筛", stage1_analysis),
    ("阶段2KMeans聚类", stage2_analysis),
    ("阶段3规则精判", stage3_analysis),
    ("出行目的识别", purpose_analysis),
    ("出行方式识别", mode_analysis),
    ("实时热力分析", heatmap_analysis),
    ("可视化", visualize_all),
]


def check_outputs():
    """课时 57 §四：逐个核对 7 个 CSV 与 6 张 PNG 是否生成。"""
    log_step(9, "结果核对：7 个 CSV + 6 张 PNG")
    missing = 0
    for name, path in config.RESULT_CSVS:
        ok = os.path.isfile(path)
        size = f"{os.path.getsize(path) / 1024:.1f} KB" if ok else "-"
        missing += 0 if ok else 1
        print(f"[{'OK' if ok else '缺失'}] {name}: {os.path.basename(path)} ({size})")
    for path in config.RESULT_FIGS:
        ok = os.path.isfile(path)
        size = f"{os.path.getsize(path) / 1024:.1f} KB" if ok else "-"
        missing += 0 if ok else 1
        print(f"[{'OK' if ok else '缺失'}] 图表: {os.path.basename(path)} ({size})")
    log(f"交付物核对：{'全部 OK' if missing == 0 else f'{missing} 个缺失'}")
    return missing


def run(sample=None, do_fig=True):
    """按顺序跑八个模块，返回缺失交付物个数；输入数据缺失时提前停住并返回 None。"""
    if sample is not None:
        config.SAMPLE_ROWS = sample            # 课时 57 §六：命令行覆盖抽样行数，0 = 全量
        log(f"SAMPLE_ROWS 覆盖为 {config.SAMPLE_ROWS}（0 表示不抽样、跑全量）")

    have, missing_files = config.check_data_files()
    if missing_files:
        # 输入缺失和「某个模块算错」不是一回事：这里没有可算的东西，继续跑只会刷
        # 八屏 traceback，所以直接停住并给出三条出路；模块级容错仍然保留在后面那层。
        log(f"数据目录 {config.DATA_DIR} 缺少 {len(missing_files)} 个输入文件："
            + "、".join(os.path.basename(p) for p in missing_files), level="ERROR")
        log("修法任选其一：① 设环境变量 TRAFFIC_DATA_DIR 指向含 6 个 CSV 的目录；"
            "② 把 6 个 CSV 放进 traffic_project/data/；"
            "③ 确认包的位置——config 会按 包内data → 仓库根data/raw → 仓库根交通时空数据分析数据 的顺序找。",
            level="ERROR")
        return None
    log(f"数据目录 {config.DATA_DIR}，输入文件 {len(have)}/6 就位")

    modules = [(n, f) for n, f in MODULES if do_fig or n != "可视化"]
    timings = {}
    for name, fn in modules:
        try:
            with Timer(name) as t:
                fn()
            timings[name] = round(t.elapsed, 2)
        except Exception as e:                 # 单模块失败不中断全流程
            log(f"模块【{name}】执行失败: {e}", level="ERROR")
            traceback.print_exc()
            log("继续执行后续模块……", level="WARN")
            timings[name] = None

    log("各模块耗时：" + " ｜ ".join(
        f"{n}={t}s" if t is not None else f"{n}=失败" for n, t in timings.items()))
    return check_outputs()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="出行/驻留识别项目一键运行")
    ap.add_argument("--sample", type=int, default=None,
                    help="整用户抽样目标行数；0 表示跑全量（默认用 config.SAMPLE_ROWS）")
    ap.add_argument("--no-fig", action="store_true", help="跳过可视化，只产出 7 个 CSV")
    args = ap.parse_args()

    log(f"项目根 {config.PROJ_DIR}")
    with Timer("全流程") as total:
        missing = run(sample=args.sample, do_fig=not args.no_fig)
    if missing is None:
        log("已提前停住，一个模块都没跑（输入数据没找到），按上面三条修好后重跑", level="ERROR")
    else:
        log(f"全部完成，总耗时 {total.elapsed:.1f} 秒")
