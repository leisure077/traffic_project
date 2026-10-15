# 课时 16：读写 CSV：无表头、编码、抽样与 dtype

## 一、read_csv 最常用参数（本项目全覆盖）

```python
pd.read_csv(
    path,
    header=None,          # 没有表头（第一行就是数据）
    names=[...],          # 手动指定列名
    usecols=[0,1,2,3],    # 只读哪几列
    dtype=str,            # 先统一按字符串读，避免长数字/前导零出问题
    encoding="gbk",       # 编码（默认utf-8）
    nrows=5,              # 只读前几行（探查用）
)
```

## 二、读"无表头"的真实文件

```python
df = pd.read_csv("原始数据.csv", header=None,
                 usecols=[0, 1, 2, 3],
                 names=["time", "imsi", "lac_id", "cell_id"],
                 dtype=str)
```

为什么 `dtype=str`：

- imsi、lac、cell 是编号不是数值，不应参与加减；
- 有些编号前导 0，当数字读会丢 0；
- 先按字符串读进来，需要时再 `pd.to_numeric`。

各静态文件的读取：

```python
station = pd.read_csv("基站经纬度数据.csv", header=None,
                      names=["lon", "lat", "laci"], dtype=str)
biz = pd.read_csv("商业区数据.csv", header=None,
                  names=["lon", "lat", "biz_name"])
poi = pd.read_csv("出行目的静态数据.csv", header=None,
                  names=["lon","lat","poi_type","poi_name"], encoding="gbk")
mode_st = pd.read_csv("出行方式静态数据.csv", header=None,
                      names=["lon","lat","mode_type","station_name","line_name"],
                      encoding="gbk")
stream = pd.read_csv("实时流数据.csv", header=None,
                     names=["time","imsi","lon","lat"], dtype={"time":str,"imsi":str})
```

> 列顺序必须和真实文件一致！讲义里曾假设"商圈名在第一列"，真实文件却是 lon 在第一列，**以真实文件为准**（见课时 06）。

## 三、数值列的安全转换

读进来的 lon/lat 是字符串（还可能有脏值），用 `to_numeric(errors='coerce')` 转成数字，转不动的变 NaN：

```python
station["lon"] = pd.to_numeric(station["lon"], errors="coerce")
station["lat"] = pd.to_numeric(station["lat"], errors="coerce")
station = station.dropna(subset=["lon", "lat", "laci"])
station = station.drop_duplicates("laci")     # 同一基站只保留一条
```

## 四、写出 CSV：to_csv

```python
df.to_csv("out.csv", index=False, encoding="utf-8-sig")
```

- `index=False`：不要把行号写进文件；
- `encoding="utf-8-sig"`：Excel 双击中文不乱码。

项目封装成 `utils.save_csv`，并自动创建不存在的目录。

## 五、关于"抽样"：为什么不能简单随机抽行（重点）

数据 53 万行，全量做聚类较慢，课堂默认抽样。但**直接随机抽行会把同一用户的轨迹抽碎**：

```python
df.sample(n=20000)   # 每人只留下约3.8%的点，大量用户不足20点，被规则9误删（实测只剩27个用户）
```

实跑教训：随机抽 2 万行后，只有最活跃的 27 个用户还能凑够 20 个点。

正确做法是**按整用户抽样**（保留被抽中用户当天的完整轨迹），将在课时 28 完整实现；最终得到约 196 个用户、17362 个点，既快又完整。

## 六、探查陌生文件的标准三连

```python
df = pd.read_csv(path, header=None, nrows=5)
print(df.shape, df.dtypes)
print(df.head())
```

中文乱码就加 `encoding="gbk"`；确认列数与含义后再正式读。

## 七、课堂练习

1. 分别用"默认方式"和"`header=None, names=...`"读基站文件，对比列名差异；
2. 读 GBK 的目的 POI 文件，打印 poi_type 的取值计数；
3. 用 `nrows=1000` 只读实时流前 1000 行看结构；
4. 思考：为什么编号列建议按字符串读？前导零丢失会对 laci 关联造成什么影响？

下一课学习行/列选择与布尔过滤——清洗规则的主力。
