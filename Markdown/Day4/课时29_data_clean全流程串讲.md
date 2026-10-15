# 课时 29：data_clean.py 全流程串讲

## 一、模块定位

`data_clean.py` 是流水线第 1 步，入口函数 `clean_data()`：读原始信令 → 9 条规则 → 产出 `cleaned_data.csv`（六列）。`main.py` 第一个调用的就是它。

## 二、导入了什么

```python
import pandas as pd
import numpy as np
from config import (RAW_DATA_FILE, STATION_DATA_FILE, CLEANED_DATA_FILE,
    TARGET_DATE, MIN_POINTS_PER_USER, SAMPLE_ROWS, RAW_TIMEZONE_OFFSET_HOURS)
from utils import log, log_step, Timer, save_csv
```

注意：**路径、阈值全部来自 config**，文件里不出现写死的路径和魔法数字。

## 三、逐段过一遍（对照 data_clean.py）

```python
log_step(1, "数据清洗（9条规则）")
```

打印醒目的步骤标题。

规则 1（读）：

```python
df = pd.read_csv(RAW_DATA_FILE, header=None, usecols=[0,1,2,3],
                 names=["time","imsi","lac_id","cell_id"], dtype=str)
```

规则 2、3（去脏、去空）：正则取反、dropna+非空串。

规则 4（时间）：毫秒戳判断 + `unit='ms', utc=True` + `Timedelta(hours=8)` + strftime。

规则 5（日期）：startswith(TARGET_DATE)。

教学抽样（整用户）：课时 28 的实现，不属于 9 条规则，日志里明确标注。

规则 6（关联）：读基站 lon,lat,laci → 去重 → 拼 laci → 左连接 → dropna。

规则 7、8、9：排序、分钟去重、≥20 点过滤。

收尾：

```python
output_cols = ["imsi","time","lac_id","cell_id","lon","lat"]
df = df[output_cols]
save_csv(df, CLEANED_DATA_FILE)
return df
```

## 四、工程细节（值得模仿的写法）

1. **每步 Timer + before/after 日志**：性能和数据变化一目了然；
2. **时间转换兼容两种格式**：用 `num.abs()>=10**12` 判断是否毫秒戳，否则按日期串解析；
3. **临时列用完即删**：`_minute`、抽样标记不留到结果；
4. **结果列严格固定为 6 列**：下游 stage1 依赖这些列；
5. **函数返回 DataFrame**：既能存盘，也能被 main 链式复用。

## 五、实际运行输出（关键行）

```
原始数据: 532828 行
去脏imsi: 532828 -> 532820（删除 8 行）
去空lac/cell: 532820 -> 532811（删除 9 行）
毫秒UTC时间戳 +8时区 转换完成，示例: 20181003004122   （约9.2秒）
保留 20181003: 532811 -> 518427（删除 14384 行）
按整用户抽样: 选中 198 个用户、19985 行
基站关联成功: 匹配后 19983 行（关联失败删除 2 行）
分钟去重: 19983 -> 17400（删除 2583 行）
每人>=20点: 17400 -> 17362 行, 196 个用户（删除 38 行）
[计时] 数据清洗 耗时 12.94 秒
```

## 六、如果结果对不上，怎么排查

| 现象 | 排查 |
| --- | --- |
| 目标日后数据骤减 | 时间戳是否漏 +8；格式串是否正确 |
| merge 后行数变多 | 基站表是否按 laci 去重 |
| 用户数只有几十 | 是否误用随机抽行（应整用户抽样） |
| lon/lat 全是 NaN | 基站列序是否按真实 lon,lat,laci；连接键类型是否一致 |
| 速度大量 inf | 是否做了分钟去重 |

## 七、课堂任务

1. 不看讲义，在草稿纸上写出 9 条规则的顺序和各自作用；
2. 在 `data_clean.py` 里给每步日志标注"对应解决的真实问题"；
3. 改一个阈值（如 MIN_POINTS_PER_USER=50）重跑，记录结果差异；
4. Code Review：找出代码里至少 3 处"防御性处理"（兼容、去重、dropna 等）。

下一课对清洗结果做数据分析，并布置 Day4 综合作业。
