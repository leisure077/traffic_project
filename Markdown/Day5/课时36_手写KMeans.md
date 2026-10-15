# 课时 36：手写 K-Means（NumPy 实现）

> 文件：`tr_stage2.py` 中的 `kmeans()`。全程不调用 sklearn。

## 一、函数签名与初始化

```python
import numpy as np

def kmeans(X, k, max_iter=100, tol=1e-6):
    n = len(X)
    if k >= n:                       # 簇数不少于点数：每个点自成一簇
        return np.arange(n, dtype=int), X.copy()
    # 1) 随机选 k 个样本作为初始中心（不重复抽样）
    centers = X[np.random.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
```

`X` 形状 `(n,2)`（n 个点的 lon,lat）。用 `np.random.choice(..., replace=False)` 从现有点里挑 k 个当初始组长。

## 二、迭代：分配 + 更新（核心）

```python
    for _ in range(max_iter):
        # 2) 分配步：每个点到所有中心的距离，取最近的簇
        dist = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        new_labels = dist.argmin(axis=1)

        # 3) 更新步：每簇成员均值为新中心；空簇保留旧中心
        new_centers = centers.copy()
        for c in range(k):
            mask = new_labels == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)

        # 中心位移，小于 tol 认为收敛
        shift = np.sqrt(((new_centers - centers) ** 2).sum())
        labels, centers = new_labels, new_centers
        if shift < tol:
            break
    return labels, centers
```

### 广播距离的形状（重点复习课时11）

```
X[:, None, :]        -> (n, 1, 2)
centers[None, :, :]  -> (1, k, 2)
相减                   -> (n, k, 2)
平方求和 axis=2       -> (n, k)   每个点到每个中心的距离
argmin(axis=1)       -> (n,)     每个点最近的簇编号
```

### 更新中心

`mask = new_labels==c` 是布尔数组，`X[mask].mean(axis=0)` 把该簇所有点在纵向上取平均，得到新中心。空簇（mask 全 False）沿用上一轮中心，避免均值对空集取导致 NaN。

## 三、收敛保证

每次"分配到最近中心 + 取簇均值"都不会让总平方误差增大，所以迭代一定会收敛（或到 max_iter 停止）。tol 控制提前结束，tol=1e-6 在经纬度尺度下已足够。

## 四、固定随机种子，结果可复现

```python
np.random.seed(RANDOM_STATE)   # 42
```

在逐用户聚类前设置一次，所有随机初始化确定，学生每次跑出同样的簇（实跑最优 K：最小 1、最大 10、平均 2.4）。

## 五、测试你写的 kmeans

```python
np.random.seed(0)
X = np.array([[0,0],[1,0],[0,1],[10,0],[11,0],[10,1]], dtype=float)
labels, centers = kmeans(X, 2)
print(labels)      # 期望大致 [0 0 0 1 1 1]
print(centers)     # 两簇中心约 (0.33,0.33)、(10.33,0.33)
```

## 六、常见 bug 与排查

| 现象 | 原因/处理 |
| --- | --- |
| 出现 NaN 中心 | 有空簇没处理，要保留旧中心 |
| 每次结果不一样 | 没固定随机种子 |
| 距离矩阵爆内存 | n 或 k 很大时改用分块；本项目按用户分组，单人点数不大 |
| 标签编号不连续 | 本实现标签即中心序号 0..k-1，天然连续 |

## 七、课堂练习

1. 不看代码默写 kmeans 的分配步广播表达式，并说出每个数组形状；
2. 造两团明显分离的点，验证聚类结果；
3. 故意删掉空簇处理，构造一个可能产生空簇的初始中心观察 NaN；
4. 对比你手写的结果与 `sklearn.cluster.KMeans`（如环境允许）的标签（允许簇编号不同，分组应一致）。

下一课解决"k 到底取几"：手写轮廓系数和邓恩指数。
