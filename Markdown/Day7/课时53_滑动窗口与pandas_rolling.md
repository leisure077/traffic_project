# 课时 53：实时热力（中2）——滑动窗口与 pandas rolling

## 一、什么是滑动窗口

"最近 10 分钟人流量"：在第 t 分钟，统计区间 `[t-9 分钟, t]`（含当前共 10 分钟）内的点数。窗口随时间向右滑动：

```
时间(分钟):  ... 10  11  12  13  14  15 ...
每分钟点数:       2   5  3   0   4   1
窗口10求和@14 = (5..14 这10分钟之和)
窗口10求和@15 = (6..15 这10分钟之和)   # 丢掉第5分钟，纳入第15分钟
```

这是时序分析的基础操作：能平滑瞬时波动、反映"当下热度"。

## 二、先按分钟 × 商圈计数

```python
stream["minute"] = stream["time_dt"].dt.floor("1min")
panel = stream.groupby(["minute","biz_name"]).size().unstack(fill_value=0)
```

- `dt.floor("1min")`：把秒归到所在分钟（12:30:45→12:30:00）；
- groupby 计数后 `unstack`：行=分钟、列=商圈、值=该分钟点数；
- `fill_value=0`：该分钟该商圈没人记 0。

## 三、补齐全天 1440 分钟

原始只在有数据的分钟有行，中间没数据的分钟缺失。不补齐，rolling 就会"跨越空档"把不相邻的分钟当连续，结果错误。

```python
day = "2018-10-03"
full_minutes = pd.date_range(f"{day} 00:00:00", periods=24*60, freq="1min")
panel = panel.reindex(full_minutes, fill_value=0)
```

`reindex` 把缺失分钟补成全 0 行，保证时间轴连续 1440 个点。

## 四、rolling 一行实现 10 分钟窗口

```python
roll = panel.rolling(HEATMAP_WINDOW_SIZE, min_periods=1).sum()
```

- 窗口大小 10：当前行与前 9 行求和；
- `min_periods=1`：一天最开始不足 10 分钟时，有几行算几行（不返回 NaN）；
- 对每个商圈列独立滑动求和。

## 五、从宽表转长表输出

可视化与结果通常要"长表"：一行一个 (时间, 商圈, 计数)。

```python
result = roll.stack().reset_index()
result.columns = ["time", "biz_name", "count"]
result["count"] = result["count"].astype(int)
result["time"] = result["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
result = result[result["count"] > 0].reset_index(drop=True)
```

- `stack` 把"商圈列"叠回到行（宽转长）；
- 去掉 count=0 的行，减小文件（只保留有人流的商圈-分钟）；
- 实跑：约 **4640 条**（命中商圈 × 它们有数据的分钟）。

## 六、手工理解 rolling

```python
s = pd.Series([2, 5, 3, 0, 4])
print(s.rolling(3, min_periods=1).sum().tolist())
# [2, 7, 10, 8, 7]
# 分别是：2 / 2+5 / 2+5+3 / 5+3+0 / 3+0+4
```

前几个位置不足窗口大小，min_periods=1 让它们按已有个数求和。

## 七、课堂练习

1. 对上面的小序列手算窗口=3 的滑动和，再用代码验证；
2. 构造一个缺分钟的 DataFrame，reindex 前后 rolling 结果有何不同；
3. 把窗口改成 5、30 分钟，观察峰值与平滑程度的变化；
4. 解释为什么"最近10分钟人数"用 sum 而不是 mean。

下一课把匹配、计数、滑窗、转表整合成 heatmap.py，并讲性能。
