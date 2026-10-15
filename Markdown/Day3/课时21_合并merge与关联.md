# 课时 21：表合并 merge（关联基站、合并簇标签）

## 一、为什么要 merge

信令原始记录里只有 lac/cell 编号，经纬度在另一张"基站表"里；阶段2算出的簇标签也要贴回到全量点上。这些都要做表连接，相当于 SQL 的 join。

```python
pd.merge(left, right, on="共同列", how="left")
```

`how` 常用：`left`（左连接，保留左表全部）、`inner`（只保留两边都有的）。

## 二、规则6：关联基站补经纬度（左连接）

```python
station = pd.read_csv("基站经纬度数据.csv", header=None,
                      names=["lon","lat","laci"], dtype=str)
station["lon"] = pd.to_numeric(station["lon"], errors="coerce")
station["lat"] = pd.to_numeric(station["lat"], errors="coerce")
station = station.dropna(subset=["lon","lat","laci"]).drop_duplicates("laci")

df["laci"] = df["lac_id"].astype(str) + "-" + df["cell_id"].astype(str)
df = df.merge(station[["laci","lon","lat"]], on="laci", how="left")
df = df.dropna(subset=["lon","lat"])
```

要点：

1. 基站表的 laci 必须唯一（`drop_duplicates`），否则连接会一对多产生重复行；
2. 左连接后匹配不到的点 lon/lat 为 NaN，再 dropna 删除；
3. 实跑：抽样数据基站匹配率接近 100%（仅 2 行匹配失败）。

## 三、阶段3：把簇标签左连接回全量点

```python
df = cleaned.merge(
    cluster[["imsi","time","cluster","is_single_cluster"]],
    on=["imsi","time"], how="left"
)
df["cluster"] = df["cluster"].fillna(-1).astype(int)
```

注意连接键是**两列组合** `["imsi","time"]`：不同用户可能在同一时间有点，单靠 time 会串号。只有阶段1判为出行的点才有 cluster，其余全部为 -1（代表驻留点）。

## 四、用 merge + indicator 找"低频点"再删除（阶段1技巧）

```python
low_freq["_is_low"] = 1
df = df.merge(low_freq, on=["imsi","lon","lat"], how="left")
df = df[df["_is_low"].isna()].drop(columns="_is_low")
```

左连接一个"低频标记表"，匹配上的点 `_is_low=1`，没匹配上的是 NaN；保留 NaN 即保留非低频点。这是一种"反连接（anti-join）"写法。

## 五、拼接多组结果：concat

逐用户聚类后，把多个小 DataFrame 拼回一个：

```python
parts = []
for imsi, g in travel.groupby("imsi"):
    parts.append(g_with_cluster)
result = pd.concat(parts, ignore_index=True)
```

## 六、常见坑

| 坑 | 后果 | 解决 |
| --- | --- | --- |
| 右表连接键不唯一 | 行数膨胀 | drop_duplicates |
| 两列同名但类型不同（str vs int） | 关联不上 | 先统一 astype(str) |
| 用 how="inner" 关联基站 | 匹配不到的用户全丢 | 本项目用 left 再 dropna |
| 只用 time 单键 | 跨用户串号 | 用 [imsi,time] 双键 |

## 七、课堂练习

1. 建一张"laci→lon/lat"小表，用左连接给信令表补经纬度；
2. 故意让右表有重复 laci，观察 merge 后行数如何膨胀；
3. 把一张含 cluster 的小表用 [imsi,time] 左连接回全量表，未命中填 -1；
4. 用 indicator/临时标记列实现"删除低频组合"的反连接。

下一课讲 pandas 时间处理——毫秒 UTC 转换、时段聚合、滑动窗口的基础。
