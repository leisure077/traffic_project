# 课时 46：出行目的（下2）——POI 最近邻匹配与 purpose.py 完整实现

## 一、读 POI 并清洗

```python
poi = read_csv_gbk(PURPOSE_POI_FILE, header=None,
                   names=["lon","lat","poi_type","poi_name"])
poi["lon"] = pd.to_numeric(poi["lon"], errors="coerce")
poi["lat"] = pd.to_numeric(poi["lat"], errors="coerce")
poi = poi.dropna(subset=["lon","lat"]).reset_index(drop=True)
```

真实列序 lon,lat,poi_type,poi_name（讲义曾假设类型在前，务必按真实顺序）。

## 二、广播算"段中心 × 全部 POI"距离矩阵

这是课时 11 广播的实战：M 个驻留段、P 个 POI，一次算出 M×P 距离（米）：

```python
poi_lon, poi_lat = poi["lon"].to_numpy(), poi["lat"].to_numpy()
c_lon = agg["center_lon"].to_numpy()[:, None]   # (M,1)
c_lat = agg["center_lat"].to_numpy()[:, None]   # (M,1)

dist_m = haversine(c_lon, c_lat,
                   poi_lon[None, :], poi_lat[None, :], unit="m")  # (M,P)

nearest_idx = dist_m.argmin(axis=1)                                  # 每段最近POI
nearest_dist = dist_m[np.arange(len(agg)), nearest_idx]
agg["nearest_poi_name"] = poi.loc[nearest_idx, "poi_name"].to_numpy()
agg["nearest_poi_type"] = poi.loc[nearest_idx, "poi_type"].to_numpy()
agg["nearest_dist_m"] = np.round(nearest_dist, 1)
agg["match_success"] = nearest_dist < PURPOSE_MATCH_THRESHOLD       # 1000m
```

要点：

- `argmin(axis=1)` 找每段最近 POI 编号；
- `dist_m[行号数组, 列号数组]` 精准取每行最近距离；
- `match_success`：最近距离 <1000m 才算有效空间证据，否则只靠时间兜底。

740 段里 683 段 1km 内有 POI（成功率高，城区 POI 密）。

## 三、综合判定并输出

```python
agg["final_purpose"] = agg.apply(decide_purpose, axis=1)
```

`decide_purpose` 即课时 45 的"居住→空间证据→时间兜底"。最后整理输出 15 列：

```
imsi, seg_id, start_time, end_time, points,
center_lon, center_lat, duration_hours, start_hour, end_hour,
final_purpose, nearest_poi_name, nearest_poi_type, nearest_dist_m, match_success
```

时间列格式化成字符串、经纬度/时长保留小数后保存 `purpose_result.csv`。

## 四、真实结果

```
共切出 740 个驻留段
POI空间匹配成功: 683/740（阈值1000m）
出行目的分布: 居住=234, 其他=229, 购物=165, 旅游=112
[计时] 出行目的识别 耗时约 0.47 秒
```

四类目的齐全。打开 CSV 能看到每段最近的具体 POI 名字（某小区/商场/景点）和距离，结果可解释。

## 五、方法的局限与可扩展方向（面试加分）

1. **最近邻过于简单**：一个点可能同时靠近商场和景点，可改为"多 POI 投票/加权/置信度"；
2. POI 数据只有类型和名称，缺少规模、营业时间，可用更丰富的 POI/用地数据；
3. 目的本质是概率推断，可用历史规律/贝叶斯/机器学习提升；
4. 当前只分析一天，结合多日规律（夜间常驻地即家）会更准。

讲项目时主动说出局限和改进方向，比只报数字更有说服力。

## 六、课堂练习

1. 单步运行 `python purpose.py`，抽查几个段的 final_purpose 与 nearest_poi_name 是否合理；
2. 把匹配阈值从 1000m 改成 300m/2000m，看匹配成功率与目的分布；
3. 统计"时间判定"与"最近POI类型"不一致的段有多少，思考如何处理；
4. 选做：把最近邻改成"500m 内出现次数最多的 POI 类型"。

## 七、Day6 小结

今天完成了出行驻留识别的最后一阶段（规则精判），并在驻留段上结合时间与 POI 空间证据识别出行目的，还修复了"购物窗口覆盖旅游"的真实逻辑缺陷。明天进入出行方式识别与商圈实时热力分析。
