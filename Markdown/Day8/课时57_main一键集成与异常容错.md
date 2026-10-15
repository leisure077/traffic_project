# 课时 57：main.py 一键集成、异常容错与结果核对

## 一、为什么需要统一入口

8 个模块可以单独运行，但要交付"一个能跑的系统"，就得有一个入口按顺序串起来，并解决：

- 顺序依赖（上游 CSV 是下游输入）；
- 某一步失败时尽量不影响后续；
- 统一计时；
- 跑完自动核对产物是否齐全。

## 二、导入 8 个主函数

```python
import config
from utils import log, Timer
from data_clean import clean_data
from tr_stage1 import stage1_analysis
from tr_stage2 import stage2_analysis
from tr_stage3 import stage3_analysis
from purpose import purpose_analysis
from mode import mode_analysis
from heatmap import heatmap_analysis
from visualize import visualize_all
```

## 三、顺序执行 + 单模块容错

```python
modules = [
    ("数据清洗", clean_data),
    ("阶段1速度粗筛", stage1_analysis),
    ("阶段2KMeans聚类", stage2_analysis),
    ("阶段3规则精判", stage3_analysis),
    ("出行目的识别", purpose_analysis),
    ("出行方式识别", mode_analysis),
    ("实时热力分析", heatmap_analysis),
    ("可视化", visualize_all),
]
for name, fn in modules:
    try:
        with Timer(name):
            fn()
    except Exception as e:
        log(f"模块【{name}】执行失败: {e}", level="ERROR")
        traceback.print_exc()
        log("继续执行后续模块……", level="WARN")
```

设计要点：

1. 把"名称 + 函数"放进列表，用循环统一调度（避免一长串重复 try）；
2. `try/except Exception` 兜底：一个模块报错时打印完整 traceback，但**继续后续模块**；
3. `with Timer(name)` 自动输出每模块耗时。

> 教学场景这种"尽量跑完"的策略便于看到哪些模块成功；生产系统往往要让关键步骤失败即停（fail fast），这是可讨论的工程权衡。

## 四、结果核对：7 CSV + 6 PNG

```python
RESULT_CSVS = [("清洗数据", config.CLEANED_DATA_FILE), ... 共7个 ...]
RESULT_FIGS = [config.FIG_STAGE3_PIE, ... 共6个 ...]

for name, path in RESULT_CSVS:
    ok = os.path.isfile(path)
    size = f"{os.path.getsize(path)/1024:.1f} KB" if ok else "-"
    print(f"[{'OK' if ok else '缺失'}] {name}: {os.path.basename(path)} ({size})")
```

真实运行结果（全部 OK）：

```
cleaned_data.csv           1250.6 KB
stage1_travel_points.csv    72.6 KB
stage2_cluster.csv          75.7 KB
stage3_label.csv          1284.5 KB
purpose_result.csv          99.3 KB
mode_result.csv             12.4 KB
heatmap_result.csv         176.7 KB
6 张 PNG 全部 OK
全部完成，总耗时约 24 秒（数据清洗约13秒、热力约4秒）
```

## 五、模块解耦的好处

每个模块"读上一步 CSV → 算 → 写下一步 CSV"：

- 可单独调试某一步（改了可视化不必重跑清洗）；
- 中间结果可检查、可复现；
- 出错时通过中间 CSV 快速定位是哪一步的问题；
- 代价是有磁盘 IO，但本项目数据量小，完全可接受（也可在 main 里直接传递 DataFrame 进一步优化）。

## 六、命令行参数化（扩展）

可引入 argparse 让运行更灵活：

```python
import argparse
p = argparse.ArgumentParser()
p.add_argument("--sample", type=int, default=20000)
p.add_argument("--no-fig", action="store_true")
args = p.parse_args()
config.SAMPLE_ROWS = args.sample
```

（作为进阶练习，不在基础要求内。）

## 七、课堂练习

1. 运行 `python main.py`，记录每模块耗时与总耗时；
2. 故意把某模块里一个文件名改错，观察容错逻辑是否继续并在结果核对里显示"缺失"；
3. 把可视化从调度列表中注释掉，确认 7 个 CSV 仍正常生成；
4. 讨论：为什么这里选择"单模块失败仍继续"，而银行交易系统通常要"失败即停"。

下一课讲性能优化与测试。
