# 课时 20：分组聚合 groupby（本项目的灵魂操作之一）

## 一、三步理解 groupby：Split - Apply - Combine

先按某列**分组**，对每组**聚合**，再**合并**成结果。

```python
df.groupby("imsi").size()                 # 每个用户多少条记录
df.groupby("imsi")["speed"].mean()        # 每个用户平均速度
df.groupby(["imsi", "lon", "lat"]).size() # 每个用户每个坐标出现几次（低频点判断）
```

## 二、规则9：过滤点数不足 20 的用户

```python
counts = df.groupby("imsi").size()
valid_users = counts[counts >= 20].index
df = df[df["imsi"].isin(valid_users)]
```

真实数据：按整用户抽样后，17400 行里有极少用户因去重跌破 20 点，过滤后 17362 行、196 个用户。

## 三、规则低频点：同用户同坐标出现次数

```python
loc_cnt = df.groupby(["imsi", "lon", "lat"]).size().reset_index(name="cnt")
low = loc_cnt[loc_cnt["cnt"] < 4]
```

阶段1用它删除"同一用户在同一坐标出现不足 4 次"的抖动/路过点（实跑 17362→11184）。

## 四、agg 一次多种聚合（驻留段/出行段都用它）

```python
agg = stay.groupby(["imsi", "stay_seg"]).agg(
    start_time=("time_dt", "min"),
    end_time=("time_dt", "max"),
    points=("time", "count"),
    center_lon=("lon", "mean"),
    center_lat=("lat", "mean"),
).reset_index()
```

命名聚合语法：`新列名=("被聚合列", "聚合函数")`。这正是 purpose.py 生成每个驻留段特征的代码。

## 五、groupby + shift：找"同一个用户的上一条记录"

阶段1算相邻点速度，要在**每个用户内部**取上一行：

```python
df["prev_lon"] = df.groupby("imsi")["lon"].shift(1)
df["prev_lat"] = df.groupby("imsi")["lat"].shift(1)
df["time_diff"] = df.groupby("imsi")["time_dt"].diff().dt.total_seconds()
```

- `shift(1)`：组内上一行（不同用户的第一行是 NaN，不会串用户）；
- `diff()`：组内与上一行的差值（时间差秒数）。

## 六、groupby + cumsum：给轨迹"切段编号"

这是本项目很巧妙的一步。先造"是否变化"的 0/1 标记，再在用户内部累加：

```python
user_change = (df["imsi"] != df["imsi"].shift(1)).astype(int)
loc_change = ((df["lon"] != df["lon"].shift(1)) |
              (df["lat"] != df["lat"].shift(1))).astype(int)
df["seg_id"] = (user_change | loc_change).groupby(df["imsi"]).cumsum()
```

含义：一旦"换用户"或"位置变化"，标记为 1，累加值就 +1，于是每一段连续相同位置得到同一个段号。实跑切出 2558 个轨迹段。

## 七、value_counts：快速计数

```python
df["label"].value_counts()                  # 0/1 各多少
purpose["final_purpose"].value_counts()     # 居住/购物/旅游/其他
df["imsi"].value_counts().head(5)           # 最活跃Top5用户（3D图用）
```

## 八、ngroups：组数

```python
df.groupby(["imsi","seg_id"]).ngroups   # 共多少段
```

## 九、课堂练习

1. 统计每个 imsi 的点数并筛出 ≥20 的用户；
2. 统计每个 imsi 在每个 (lon,lat) 的次数，找出 <4 的低频组合；
3. 造一个按用户排序的小表，用 groupby+shift 取上一条坐标和时间差；
4. 用"变化标记+cumsum"实现轨迹切段，并解释为什么要 `.groupby(imsi).cumsum()`。

下一课学习 merge——把经纬度、簇标签关联到主表。
