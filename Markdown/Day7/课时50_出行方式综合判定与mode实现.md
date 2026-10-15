# 课时 50：出行方式（下）——综合判定与 mode.py 完整实现

## 一、综合裁决规则（站点证据优先，速度兜底）

```python
if sub_pts >= SUBWAY_MIN_POINTS:       # >=2 个点近地铁站
    final_mode = "地铁"
elif bus_pts >= BUS_MIN_POINTS:        # >=3 个点近公交站
    final_mode = "公交"
elif avg_speed > MODE_BUS_SPEED:       # 平均速度>20
    final_mode = "小汽车"
else:
    final_mode = "其他"
```

设计逻辑：

1. **先地铁**：地铁站点稀疏、证据强，≥2 点近地铁站优先认地铁；
2. **再公交**：公交站密集，需要 ≥3 点才认；
3. **再小汽车**：没有轨交/公交证据但速度快（>20），判小汽车；
4. **其他**：速度不快、也不贴站点——步行、骑行或证据不足。

为什么站点证据排在速度前面：地铁和小汽车速度区间重叠，只有空间证据能把它们分开；而速度对"公交 vs 慢行"仍有区分力，作为兜底。

## 二、把每段结果收集成表

逐段 append 字典，最后 `pd.DataFrame`：

```python
records.append({
    "imsi": imsi, "seg_id": seg, "points": len(g),
    "start_time": format_time(first["time_dt"]),
    "end_time": format_time(last["time_dt"]),
    "start_lon": round(float(first["lon"]),6), "start_lat": round(float(first["lat"]),6),
    "end_lon": round(float(last["lon"]),6),   "end_lat": round(float(last["lat"]),6),
    "distance_km": round(distance_km, 3),
    "avg_speed": round(avg_speed, 2),
    "subway_points": sub_pts, "bus_points": bus_pts,
    "final_mode": final_mode,
})
result = pd.DataFrame(records)
save_csv(result, MODE_RESULT_FILE)
```

输出 14 列，既保留最终结论，也保留速度、邻近点数等**可解释证据**，便于答辩时回答"为什么这段判地铁"。

## 三、真实结果

```
地铁站 68 个，公交站 900 个
逐段判定（共551段）
有效出行段 97（丢弃单点段 454、无有效速度段 0）
出行方式分布: 小汽车=37, 其他=29, 公交=28, 地铁=3
```

四类齐全。地铁偏少与数据特点有关：地铁段需要连续多个点贴近站，而多数出行段只有 2~3 个点、且样本中地铁站点覆盖有限；这是结果可以如实讨论的点，而不是要"美化"成均匀分布。

## 四、结果可信度怎么讲

- 每段判定都能追溯到 avg_speed、subway_points、bus_points 三个证据字段；
- 阈值（12/20 km/h、500m、2/3 点）集中在 config，可做敏感性实验；
- 局限：信令稀疏导致大量单点段无法判方式；直线距离、基站误差也会影响。可扩展方向：结合线路缓冲区、地图路径匹配、多日数据。

## 五、课堂练习

1. 运行 `python mode.py`，抽查 3 段，对照证据字段验证 final_mode；
2. 把 `SUBWAY_MIN_POINTS` 改成 1，观察地铁段数量变化并解释误判风险；
3. 交换判定顺序（先速度后站点），看地铁/小汽车数量如何变化，说明为什么站点优先；
4. 统计被丢弃的单点段占比，讨论信令稀疏对方式识别的影响。

## 六、阶段性小结

到这里，基于手机信令的"出行/驻留→目的→方式"个人出行链已完成。下面进入面向**群体、实时**的分析：用实时流数据做商圈 10 分钟人流热力。
