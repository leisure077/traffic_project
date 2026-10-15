# 课时 14：utils 工具模块整合实战（Day2 综合课）

## 一、学习目标

- 把前两天的 Haversine、时间解析、读写、日志、计时器整合进 `utils.py`；
- 学会运行模块自测 `python utils.py`；
- 能在后续业务代码里熟练调用这些工具。

## 二、utils.py 全貌（逐块讲解）

### 1. 距离：haversine（课时 13 已手写）

### 2. 时间解析与格式化

```python
import pandas as pd

def parse_time(time_str, fmt="%Y%m%d%H%M%S"):
    return pd.to_datetime(time_str, format=fmt, errors="coerce")

def format_time(dt, fmt="%Y-%m-%d %H:%M:%S"):
    return dt.strftime(fmt)
```

`errors="coerce"` 表示解析失败的变成 `NaT`（时间里的空值），而不是直接报错中断。

格式符含义：`%Y`年 `%m`月 `%d`日 `%H`时 `%M`分 `%S`秒。所以 `20181003123000 → 2018-10-03 12:30:00`。

### 3. 读写 CSV（自动处理中文编码/目录）

```python
def read_csv(path, dtype=None, encoding="utf-8", **kw):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path, dtype=dtype, encoding=encoding, **kw)

def read_csv_gbk(path, dtype=None, **kw):
    return read_csv(path, dtype=dtype, encoding="gbk", **kw)

def save_csv(df, path, index=False, encoding="utf-8-sig", **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index, encoding=encoding, **kw)
```

设计要点：

- 文件不存在时抛出**清晰的中文错误**，而不是 pandas 那串难懂的异常；
- 单独提供 `read_csv_gbk`，记住两个静态数据是 GBK；
- `save_csv` 用 `utf-8-sig`（Excel 友好）并自动 `makedirs`，避免"目录不存在"报错。

### 4. 日志：log / log_step

```python
import time
def log(msg, level="INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")

def log_step(n, name):
    print(f"\n{'='*60}\n步骤{n}：{name}\n{'='*60}")
```

带时间戳的输出便于复盘每一步耗时；`log_step` 打印醒目的分隔标题。

### 5. 打印表概况

```python
def print_df_stats(df, name="DataFrame"):
    print(f"{name}: 行数={len(df)}, 列={list(df.columns)}")
    print(df.head(3))
```

数据处理中要"**频繁看数据**"，不要闭着眼睛写。

### 6. 计时器（上下文管理器）

```python
class Timer:
    def __init__(self, name="任务"):
        self.name = name
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        print(f"  [计时] {self.name} 耗时 {time.perf_counter()-self.t0:.2f} 秒")
```

用法：

```python
with Timer("数据清洗"):
    ...   # 退出缩进自动打印耗时
```

实跑参考：数据清洗约 12.9 秒（其中毫秒时间戳转换约 9.2 秒，是全流程最耗时的一步）。

## 三、模块自测的写法

`utils.py` 末尾：

```python
if __name__ == "__main__":
    print(haversine(123.4119, 41.8052, 123.4370, 41.8033))   # 约2.1km
    print(format_time(parse_time("20181003123000")))
    with Timer("自测"):
        time.sleep(0.2)
```

运行 `python utils.py`，能看到距离、时间、计时三项自测输出。

## 四、NumPy 知识在本项目的落点回顾

| NumPy 技能 | 用在哪个模块 |
| --- | --- |
| 向量化四则/三角 | haversine、速度 |
| 布尔索引 | 速度阈值、异常速度过滤、命中商圈 |
| `argmin(axis=1)` | 最近 POI / 站点 / 商圈 |
| 广播 `[:,None]` | 一点对多点距离矩阵、K-Means 点到中心 |
| `mean(axis=0)` | 簇中心、驻留段中心 |
| `np.unique / value_counts` | 簇数、单点簇统计 |
| 分批处理 | 25.6 万流点匹配商圈 |

## 五、动手任务

1. 运行 `python utils.py`，确认三项自测通过；
2. 在 `utils.py` 里新增一个函数 `kmh(dist_km, seconds)`：用距离千米和秒数算 km/h，并自测；
3. 用 `Timer` 对比"for 循环算 100 万次正弦"与"`np.sin` 向量化"的耗时；
4. 思考：为什么保存结果用 `utf-8-sig`，而读取静态数据要用 `gbk`？

## 六、Day2 小结

这一天你掌握了 NumPy 的数组、向量化、布尔索引、轴、广播，并手写了球面距离 Haversine，理解了经纬度近似的取舍。Day3 进入本项目使用最频繁的 pandas，把这些数值计算装进"表格处理"的完整工作流。
