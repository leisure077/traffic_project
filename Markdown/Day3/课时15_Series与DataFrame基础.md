# 课时 15：Pandas 的 Series 与 DataFrame

## 一、Pandas 在本项目中的地位

数据清洗、分组、连接、时间处理，几乎全是 pandas。可以说：**NumPy 负责算，pandas 负责"带着列名和索引组织数据并批量处理"**。

- `Series`：一列数据（带索引）；
- `DataFrame`：一张二维表（由多个 Series 组成）。

## 二、创建 Series / DataFrame

```python
import pandas as pd

s = pd.Series([8.0, 17.0, 30.0], name="speed")
print(s.mean(), s.max())

df = pd.DataFrame({
    "imsi": ["u1", "u2", "u3"],
    "speed": [8.0, 17.0, 30.0],
    "label": [0, 1, 1],
})
print(df)
```

## 三、查看数据（处理真实数据第一步永远是"看"）

```python
df.head()        # 前5行
df.tail()        # 后5行
df.shape         # (行数, 列数)
df.columns       # 列名
df.dtypes        # 每列类型
df.describe()    # 数值列统计
df.info()        # 类型+非空数量+内存
```

## 四、选取列与新增列

```python
df["speed"]                 # 取一列 -> Series
df[["imsi", "speed"]]       # 取多列 -> DataFrame

df["is_fast"] = df["speed"] > 20          # 新增布尔列
df["speed_ms"] = df["speed"] / 3.6        # km/h -> m/s
```

项目里新增列的例子：

```python
df["laci"] = df["lac_id"].astype(str) + "-" + df["cell_id"].astype(str)
df["_minute"] = df["time"].str[:12]
```

## 五、用真实数据建一个迷你 DataFrame

清洗后的每条记录最终长这样（对应 `cleaned_data.csv`）：

```python
cleaned = pd.DataFrame({
    "imsi": ["u1", "u1", "u2"],
    "time": ["20181003080100", "20181003080500", "20181003090000"],
    "lac_id": ["16481", "16481", "16482"],
    "cell_id": ["536531", "536531", "536532"],
    "lon": [123.40, 123.40, 123.45],
    "lat": [41.80, 41.80, 41.85],
})
print(cleaned.shape, cleaned.columns.tolist())
```

## 六、index（索引）是什么

每行有个索引，默认 0,1,2…。布尔过滤、合并、分组都依赖它。一般不用管，但要知道：

- 过滤后索引可能不连续，需要时用 `.reset_index(drop=True)` 重排；
- `df.loc[idx, col]` 按"索引标签+列名"取值，项目里给某些行打标签会用到。

## 七、小练习

1. 用字典创建一个含 imsi、speed、label 三列的 5 行 DataFrame；
2. 新增一列 `is_travel = speed > 8`；
3. 打印它的 shape、dtypes、describe；
4. 用 `df[["imsi","is_travel"]]` 只看这两列。

下一课学习读 CSV 的各种正确姿势——无表头、指定列、GBK、抽样。
