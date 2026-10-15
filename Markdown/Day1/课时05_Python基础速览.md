# 课时 05：Python 基础速览（为本项目量身复习）

> 同学们已有 Python 基础，本课不系统讲语法，只复习本项目**高频用到**的写法。建议边看边在 Python 里敲。

## 一、容器：list / dict / tuple

```python
points = [12, 35, 8]          # 列表，可变
points.append(9)
print(points[0], points[-1])  # 12 9，-1 表示最后一个

d = {"imsi": "u001", "speed": 18.5}   # 字典：键值对
print(d["imsi"], d.get("flag", 0))    # get 取不存在的键时返回默认值0

coords = (123.4, 41.8)        # 元组，不可变，常用于“一组坐标”
```

项目里切轨迹段、收集结果记录，大量使用"空列表 + append + 最后转 DataFrame"：

```python
records = []
records.append({"imsi": "u1", "speed": 18.5})
result = pd.DataFrame(records)
```

## 二、循环、判断与推导式

```python
for i, v in enumerate(["a", "b"]):   # 同时取下标和值
    print(i, v)

scores = [3, 15, 7, 40]
fast = [s for s in scores if s > 8]            # 列表推导式
print(fast)                                     # [15, 40]

name = {"地铁": "sub", "公交": "bus"}.get("地铁", "other")
```

字典推导式在统计个数时很常用：

```python
cnt = {}
for lab in labels:
    cnt[lab] = cnt.get(lab, 0) + 1
```

## 三、函数与默认参数

```python
def haversine(lon1, lat1, lon2, lat2, unit="km"):
    ...
    return d * 1000 if unit == "m" else d

haversine(123.4, 41.8, 123.5, 41.9)          # 默认千米
haversine(123.4, 41.8, 123.5, 41.9, "m")     # 米
```

要点：**小步封装**，一个函数只干一件事；阈值通过参数传入，不写死。

## 四、字符串与正则

```python
s = "20181003123000"
print(s[:8], s[:12])     # '20181003' 前8位=日期；前12位=到分钟
print(s.startswith("20181003"))   # True，用于筛选目标日
```

正则用于识别脏 imsi（含 `#`、`*`、空格）：

```python
import re
bool(re.search(r"[#* ^\s]", "ab*1"))   # True，含特殊字符
```

pandas 里对应的是 `Series.str.contains(r"[#* ^\s]", regex=True)`。

## 五、异常处理

```python
try:
    fn()
except Exception as e:
    print("出错了：", e)
```

`main.py` 对每个模块 try/except：一个模块失败时打印错误并**继续执行后续模块**，不会整个程序崩掉。

## 六、模块导入

同一个文件夹下，文件之间可以直接导入：

```python
from config import SPEED_THRESHOLD
from utils import haversine, log, Timer
```

`if __name__ == "__main__":` 保证"被导入时不执行、直接运行时才执行"。

## 七、lambda 与 apply（pandas 里高频）

```python
def level(speed):
    return "快" if speed > 20 else "慢"

df["level"] = df["speed"].apply(level)          # 对每一行调用
df["hour"] = df["time_dt"].apply(lambda x: x.hour)
```

## 八、布尔与按位逻辑（容易踩坑）

pandas 里对整列做"且/或"不能用 `and/or`，要用 `& / |`，并且每个条件要加括号：

```python
ok = (df["speed"] > 0) & (df["speed"] < 200)
day = (df["hour"] < 6) | (df["hour"] >= 20)
```

## 九、时间与类型（先有印象，Day3 细讲）

```python
import numpy as np
arr = np.array([1, 2, 3])          # numpy 数组
print(arr.mean(), arr.max())

import pandas as pd
s = pd.Series([1, None, 3])        # 类似“一列”
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
```

## 十、课堂自测（能写出来就算过关）

1. 用列表推导式，从 `[2, 9, 11, 25, 7]` 中筛出 ≥ 10 的数；
2. 写一个函数 `is_night(hour)`：当 `hour` 在 0~6 或 20~23 时返回 `True`；
3. 写一个循环，统计列表 `["地铁","公交","地铁"]` 中各类别出现次数（用字典）；
4. 说出 `"20181003123000"[:12]` 的结果，以及它为什么能用于"分钟去重"。

参考答案见 `utils.is_night`、`tr_stage1` 中的分组统计；第 4 题答案：`201810031230`（精确到分钟），同一用户同一分钟相同即重复。
