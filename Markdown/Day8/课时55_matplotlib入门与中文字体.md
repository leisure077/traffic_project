# 课时 55：Matplotlib 入门与中文显示

## 一、为什么用 Matplotlib

它是 Python 最基础、可控性最强的绘图库，pandas 的画图也基于它。本项目 6 张图都用它，并设置无界面后端 `Agg`，在没有显示器的服务器上也能存图。

## 二、最小可用模板

```python
import matplotlib
matplotlib.use("Agg")          # 不弹窗，直接存文件（放在 pyplot 导入前）
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]     # 中文
plt.rcParams["axes.unicode_minus"] = False       # 负号正常显示

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([0, 1, 2], [3, 1, 4])
ax.set_title("标题"); ax.set_xlabel("x"); ax.set_ylabel("y")
fig.savefig("demo.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

## 三、中文字体的自动回退（项目实际写法）

不同机器不一定有 SimHei，代码先检测已安装字体，按 SimHei→微软雅黑→宋体→楷体 回退：

```python
import matplotlib.font_manager as fm
avail = {f.name for f in fm.fontManager.ttflist}
for f in ("SimHei", "Microsoft YaHei", "SimSun", "KaiTi"):
    if f in avail:
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
```

若图里中文仍是方框，说明这些中文字体都缺失，安装 SimHei 或把一个中文字体文件放进项目并 `font_manager.fontManager.addfont(...)` 注册。

## 四、最常用 API（本项目用到的）

| 图形 | API | 对应图 |
| --- | --- | --- |
| 折线 | `ax.plot(x, y, marker=, label=)` | 24 小时折线、3D 轨迹 |
| 柱状 | `ax.bar(labels, values, color=)` | 目的/方式分布 |
| 饼图 | `ax.pie(values, labels, autopct, colors)` | 出行/驻留占比 |
| 热力 | `ax.imshow(matrix, cmap='YlOrRd')` + `fig.colorbar` | 商圈×小时 |
| 3D | `fig.add_subplot(111, projection='3d')` | Top5 用户轨迹 |

通用设置：`set_title / set_xlabel / set_ylabel / set_xticks / legend / grid`，保存统一 `savefig(path, dpi=150, bbox_inches='tight')`。

## 五、给柱状图加"数值+百分比"标签（目的/方式图）

```python
bars = ax.bar(idx, values, color=colors)
for b, v in zip(bars, values):
    pct = f"{v / total * 100:.1f}%"
    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
            f"{v}\n{pct}", ha="center", va="bottom")
```

## 六、imshow 画热力矩阵的数据准备

需要一个"行=商圈、列=24小时"的二维矩阵：

```python
# heat 长表(time,biz_name,count) -> 矩阵
heat["hour"] = pd.to_datetime(heat["time"]).dt.hour
pivot = heat.groupby(["biz_name","hour"])["count"].max().unstack(fill_value=0)
pivot = pivot.reindex(columns=range(24), fill_value=0)
top = pivot.max(axis=1).sort_values(ascending=False).head(10)
M = pivot.loc[top.index].to_numpy()
ax.imshow(M, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(len(M))); ax.set_yticklabels(top.index)
```

这里每格取该小时内"10 分钟窗口计数"的最大值（峰值），刻画该商圈这一小时最挤的 10 分钟。

## 七、课堂练习

1. 配置好中文字体，画一条带中文标题的折线并存 PNG；
2. 用真实 mode_result.csv 画一个方式分布柱状图，加数值标签；
3. 故意不设 `axes.unicode_minus=False`，观察负号显示；
4. 把 dpi 改成 72/300，比较图片清晰度与文件大小。

下一课完整实现 6 张图。
