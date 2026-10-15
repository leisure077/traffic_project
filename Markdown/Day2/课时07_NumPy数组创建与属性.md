# 课时 07：NumPy 数组的创建与属性

## 一、为什么本项目离不开 NumPy

后面要做的距离计算、K-Means、轮廓系数，本质都是"**一大批数同时做加减乘除**"。用 Python 原生列表 + for 循环能写，但慢。NumPy 的 `ndarray`（多维数组）把同类数据放在连续内存里，支持**整列运算**，又快又简洁。

## 二、创建数组

```python
import numpy as np

a = np.array([1, 2, 3])                 # 从列表创建
b = np.array([[1, 2], [3, 4]])          # 二维数组（像一张小表）
z = np.zeros(3)                         # [0., 0., 0.]
o = np.ones((2, 3))                     # 2行3列全1
r = np.arange(0, 10, 2)                 # [0 2 4 6 8]
print(np.linspace(0, 1, 5))             # 0到1等距5个数
```

项目里经纬度常从 pandas 列转成数组：

```python
lon = df["lon"].to_numpy()              # pandas Series -> numpy数组
X = g[["lon", "lat"]].to_numpy(float)   # 两列 -> 形状 (n,2) 的坐标矩阵
```

## 三、核心属性：shape、ndim、dtype、size

```python
X = np.array([[123.4, 41.8],
              [123.5, 41.9],
              [123.6, 42.0]])
print(X.shape)   # (3, 2)：3个点、每个点2个坐标
print(X.ndim)    # 2：二维
print(X.size)    # 6：元素总数
print(X.dtype)   # float64：元素类型
```

> **最重要的是 `shape`**。K-Means 里坐标矩阵是 `(n, 2)`，距离矩阵是 `(n, n)`。出 bug 时第一步永远是打印 shape。

## 四、形状变换（reshape）与广播雏形

```python
a = np.arange(6)
print(a.reshape(2, 3))      # 变成2行3列
print(a.reshape(-1, 1))     # -1 自动推断 -> 6行1列
```

`reshape(-1, 1)` 在"一点对多点"算距离时反复用到（课时 11、13 细讲）。

## 五、索引与切片

```python
a = np.array([10, 20, 30, 40, 50])
print(a[0], a[-1], a[1:4])     # 10 50 [20 30 40]

X = np.array([[1, 2], [3, 4], [5, 6]])
print(X[0, 1])      # 2：第0行第1列
print(X[:, 0])      # [1 3 5]：所有行的第0列（所有经度）
print(X[1])         # [3 4]：第1行（第1个点）
```

## 六、用项目坐标做练习

```python
# 3个点的(经度,纬度)
X = np.array([[123.40, 41.80],
              [123.41, 41.81],
              [123.42, 41.79]])
print(X.shape)        # (3, 2)
print(X[:, 0])        # 所有经度 [123.40 123.41 123.42]
print(X[0])           # 第一个点 [123.4 41.8]
print(X[1:])          # 后两个点
```

## 七、常见错误

| 错误写法 | 问题 | 正确 |
| --- | --- | --- |
| `X[1,2]` 但 X 只有 2 列 | 下标越界 | 先看 shape |
| 列表直接做矩阵运算 `[[1,2]]*2` | 这是复制不是乘法 | 用 `np.array` |
| 形状对不上还硬算 | 广播报错 | 打印 shape，用 reshape |

## 八、课堂练习

1. 用 `np.array` 创建 4 个点的坐标矩阵，打印 shape，并取出所有纬度；
2. 创建一个 `(5, 3)` 的全 1 数组，再 reshape 成 `(-1, 1)`，打印新 shape；
3. 从一个 pandas 两列 DataFrame 用 `to_numpy(float)` 得到 `(n,2)` 矩阵。

下一课学习 NumPy 最核心的"向量化"思想——为什么不要写 for 循环。
