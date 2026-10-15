# 课时 13：手写 Haversine 球面距离

## 一、为什么是 Haversine

地球近似一个半径 R≈6371 km 的球体。已知两点经纬度，求球面最短距离（大圆距离），常用 **Haversine 公式**。本项目所有需要真实米数的地方都用它：速度计算、目的 1000m、方式 500m、商圈 1000m。

## 二、公式拆解（不怕，一步步来）

把经纬度先转弧度，设：

```
Δlat = lat2 - lat1
Δlon = lon2 - lon1
a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
c = 2·arcsin(√a)
距离 d = R · c        （千米）
```

记忆法：先算"**纬度差的贡献**"，再算"**经度差乘以两地纬度余弦的贡献**"，开方取弧长，乘地球半径。

## 三、代码实现（支持标量，也支持数组广播）

```python
import numpy as np

def haversine(lon1, lat1, lon2, lat2, unit="km"):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    d = R * c
    return d * 1000.0 if unit == "m" else d
```

要点：

1. `map(np.radians, [...])` 一次性把四个量转弧度；
2. 全程用 NumPy 函数，所以传单个数字（标量）或数组都行；
3. `unit="m"` 时乘 1000 返回米。

## 四、验证（项目自测用例）

```python
# 同一点距离必须是 0
print(haversine(123.4, 41.8, 123.4, 41.8, unit="m"))   # 0.0

# 太原街 -> 中街，沈阳两个相距约2km的商圈
print(haversine(123.4119, 41.8052, 123.4370, 41.8033))  # 约 2.1 km
```

跑 `python utils.py` 会自动打印这些自测结果。

## 五、数组广播：一次算"一点对多个 POI"

因为函数内部全是 NumPy 运算，直接传入不同形状即可广播：

```python
center_lon = np.array([123.40, 123.42])[:, None]   # (2,1)
center_lat = np.array([41.80, 41.82])[:, None]     # (2,1)
poi_lon = np.array([123.40, 123.41, 123.43])[None, :]  # (1,3)
poi_lat = np.array([41.80, 41.81, 41.79])[None, :]
d = haversine(center_lon, center_lat, poi_lon, poi_lat, unit="m")
print(d.shape)   # (2,3)：2个中心到3个POI的距离矩阵
```

这正是 `purpose.py / mode.py / heatmap.py` 做最近邻匹配的基础。

## 六、用 Haversine 算速度（tr_stage1 的关键）

相邻两点距离（km）除以时间差（小时）就是速度：

```python
dist_km = haversine(prev_lon, prev_lat, lon, lat, unit="km")
speed_kmh = dist_km * 3600.0 / time_diff_seconds
```

`time_diff_seconds` 是秒，`×3600` 换成"每小时"。之后 `speed > 8` 粗判出行，`speed > 200` 视为基站跳跃异常。

## 七、课堂练习

1. 手算（或代码验证）：同一经纬度两点距离是否为 0；
2. 用函数计算两个你熟悉地点的距离，与地图软件结果对比；
3. 构造 2 个中心、4 个 POI，用广播得到 `(2,4)` 距离矩阵并用 `argmin(axis=1)` 找最近 POI；
4. 解释为什么公式里经度项要乘 `cos(lat1)cos(lat2)`（提示：高纬经线更密）。

下一课把 Haversine、时间、读写、日志、计时器整合进 `utils.py` 并完成自测。
