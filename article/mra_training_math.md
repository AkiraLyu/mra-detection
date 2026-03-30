# `mra.py` 训练流程与数学原理

本文只分析当前代码 [`mra.py`](./mra.py) 的真实执行路径，并把每一步对应的数学含义写清楚。重点覆盖：

- 数据如何变成训练样本
- 单个 batch 在训练时如何被进一步遮挡
- AGF-ADNet 前向传播每个模块在计算什么
- 损失函数为什么这样设计
- 训练结束后如何把重构误差变成异常判定

---

## 1. 总体流程

`train()` 的主路径可以概括为：

1. 从 `data/train` 和 `data/test` 读入 CSV，并记录缺失掩码
2. 用 `0` 临时填补缺失值，再用训练集统计量做标准化
3. 按长度 `S=60` 构造滑动窗口，得到窗口样本 `Xtr, Mtr`
4. 对每个随机种子训练或加载一个 `AGF_ADNet`
5. 用训练好的模型计算每个窗口的重构误差，作为异常分数
6. 对 3 个 seed 的分数做平均，再做阈值化、EWMA 平滑和持续段过滤

从数学上看，它是一个“带图学习、时频双分支、Transformer 重构头”的自监督重构模型，训练目标不是直接分类，而是：

$$
\text{让模型学会在部分观测被遮挡时重构原始窗口。}
$$

异常检测阶段再利用“异常样本更难被重构”这一假设：

$$
\text{anomaly score} \uparrow \quad \Longleftrightarrow \quad \text{reconstruction error} \uparrow
$$

---

## 2. 符号约定

设：

- 原始时间序列长度为 $T$
- 变量个数为 $N$
- 窗口长度为 $S=60$
- batch 大小为 $B$

定义：

- 原始数据矩阵：$\mathbf{D} \in \mathbb{R}^{T \times N}$
- 缺失掩码：$\mathbf{M} \in \{0,1\}^{T \times N}$
  - `1` 表示缺失
  - `0` 表示观测到
- 一个窗口样本记为：
  $$
  \mathbf{X} \in \mathbb{R}^{B \times S \times N}, \qquad
  \mathbf{M} \in \{0,1\}^{B \times S \times N}
  $$

代码中多数张量遵循 `(batch, seq_len, num_features)`，也就是 `(B, S, N)`。

---

## 3. 数据准备

对应代码：`mra.py` 第 32-89 行、第 617-655 行。

### 3.1 读取数据与缺失掩码

`DatasetBuilder.load_dir()` 会把多个 CSV 按时间拼接：

$$
\mathbf{D} =
\begin{bmatrix}
\mathbf{D}^{(1)} \\
\mathbf{D}^{(2)} \\
\cdots \\
\mathbf{D}^{(K)}
\end{bmatrix}
$$

同时构造缺失掩码：

$$
M_{t,n} =
\begin{cases}
1, & D_{t,n} \text{ 是 NaN} \\
0, & D_{t,n} \text{ 是观测值}
\end{cases}
$$

### 3.2 零填补后标准化

`train()` 中采用的是：

```python
train_filled = np.nan_to_num(train_data, nan=0.0)
test_filled = np.nan_to_num(test_data, nan=0.0)
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_filled)
test_data_scaled = scaler.transform(test_filled)
```

也就是先把缺失值临时替换成 `0`，再按训练集统计量做 z-score 标准化：

$$
\mu_n = \frac{1}{T_{\text{train}}} \sum_{t=1}^{T_{\text{train}}} \tilde{D}_{t,n}, \qquad
\sigma_n = \sqrt{
\frac{1}{T_{\text{train}}}
\sum_{t=1}^{T_{\text{train}}} (\tilde{D}_{t,n} - \mu_n)^2
}
$$

$$
\hat{D}_{t,n} = \frac{\tilde{D}_{t,n} - \mu_n}{\sigma_n}
$$

其中 $\tilde{D}$ 表示“NaN 已被 0 替换后的数据”。

这一步有一个很重要的实现细节：

- 原始缺失位在标准化之后一般 **并不等于 0**
- 因为它们先被当作数值 `0` 参与了变换，结果会变成
  $$
  \frac{0 - \mu_n}{\sigma_n}
  $$
- 真正把输入中的缺失位重新置零，是在后面的 `apply_missing_mask()` 中完成的

### 3.3 滑动窗口

窗口由 `create_windows()` 构造。对每个时间点 $i$，取最近 `S=60` 个时刻组成一个窗口：

$$
\mathbf{X}^{(i)} =
\begin{bmatrix}
\hat{\mathbf{D}}_{i-S+1} \\
\hat{\mathbf{D}}_{i-S+2} \\
\cdots \\
\hat{\mathbf{D}}_i
\end{bmatrix}
\in \mathbb{R}^{S \times N}
$$

对应的掩码窗口为：

$$
\mathbf{M}^{(i)} =
\begin{bmatrix}
\mathbf{M}_{i-S+1} \\
\mathbf{M}_{i-S+2} \\
\cdots \\
\mathbf{M}_i
\end{bmatrix}
\in \{0,1\}^{S \times N}
$$

当前代码从 `i = 99` 开始取窗口，因此在默认配置下：

- `seq_len = 60`
- `i >= 99 > 60`

所以 `create_windows()` 里那段“左侧复制首行做 padding”的分支，在当前默认训练流程里实际上不会触发。

---

## 4. 单个 seed 的训练目标

对应代码：`mra.py` 第 513-581 行。

### 4.1 自监督遮挡

训练不是直接用原窗口做自编码，而是先在“原本可观测的位置”上随机再遮掉一部分。

先定义原始可观测掩码：

$$
\mathbf{O} = 1 - \mathbf{M}
$$

然后对观测位做随机 dropout：

$$
\mathbf{R}_{b,s,n} \sim \text{Bernoulli}(0.1), \qquad \mathbf{R} \odot \mathbf{M} = 0
$$

代码里：

```python
observed = ~m.bool()
rand_drop = (torch.rand_like(x) < 0.1) & observed
```

于是：

- `rand_drop` 只会落在原本观测到的位置
- `target_mask = rand_drop.float()` 表示这些位置是训练时真正要重构的监督目标

如果某个 batch 恰好没有采样到任何随机遮挡位，代码会退化成“监督所有观测位”：

$$
\mathbf{T} =
\begin{cases}
\mathbf{R}, & \mathbf{R}\neq 0 \\
\mathbf{O}, & \mathbf{R}=0
\end{cases}
$$

### 4.2 构造模型输入

训练输入掩码：

$$
\mathbf{M}^{\text{in}} = \max(\mathbf{M}, \mathbf{R})
$$

对应代码：

```python
m_input = m.clone()
m_input[rand_drop] = 1.0
```

再把输入中这些缺失位置清零：

$$
\mathbf{X}^{\text{in}} = \mathbf{X} \odot (1 - \mathbf{M}^{\text{in}})
$$

这一步由：

```python
x_input = apply_missing_mask(x, m_input)
```

实现。

所以训练本质上是一个 masked reconstruction 问题：

$$
\mathbf{X}^{\text{in}} \xrightarrow{\text{model}} \mathbf{X}^{\text{rec}}
$$

模型必须从剩余上下文中恢复被隐藏的信息。

---

## 5. AGF-ADNet 前向传播

对应代码：`mra.py` 第 95-449 行。

整体可以写成：

$$
(\mathbf{X}^{\text{rec}}, \mathbf{A}, \mathbf{X}^{\text{imp}})
= f_\theta(\mathbf{X}^{\text{in}}, \mathbf{M}^{\text{in}})
$$

其中：

- $\mathbf{A} \in \mathbb{R}^{B \times N \times N}$ 是自适应邻接矩阵
- $\mathbf{X}^{\text{imp}} \in \mathbb{R}^{B \times S \times N}$ 是时频融合后的插补结果
- $\mathbf{X}^{\text{rec}} \in \mathbb{R}^{B \times S \times N}$ 是最终重构结果

### 5.1 图学习模块 `GraphLearner`

对应代码：第 95-177 行。

#### 5.1.1 每个变量节点的动态统计

对每个变量 $n$，代码从一个窗口里抽取 4 个动态特征：

1. 均值
2. 标准差
3. 最近一次观测值
4. 缺失率

令窗口内第 $n$ 个变量的序列为 $\mathbf{x}_{:,n}$，对应观测指示为 $\mathbf{o}_{:,n}=1-\mathbf{m}_{:,n}$，则：

$$
\mu_n = \frac{\sum_{s=1}^{S} x_{s,n} o_{s,n}}{\sum_{s=1}^{S} o_{s,n}}
$$

$$
\sigma_n = \sqrt{
\frac{\sum_{s=1}^{S} (x_{s,n} - \mu_n)^2 o_{s,n}}
{\sum_{s=1}^{S} o_{s,n}} + 10^{-6}
}
$$

最近观测值：

$$
\ell_n = x_{s^\star,n}, \qquad
s^\star = \max \{ s \mid o_{s,n}=1 \}
$$

缺失率：

$$
\rho_n = \frac{1}{S} \sum_{s=1}^{S} m_{s,n}
$$

于是节点动态特征为：

$$
\mathbf{d}_n = [\mu_n, \sigma_n, \ell_n, \rho_n]
$$

代码还给每个节点引入一个可学习静态向量 $\mathbf{s}_n$，两者拼接后经过 MLP：

$$
\mathbf{h}_n = \phi([\mathbf{d}_n, \mathbf{s}_n])
$$

#### 5.1.2 邻接先验

代码里没有人工给图，而是学习每个节点坐标 $\mathbf{c}_n$，再用反距离构造先验：

$$
p_{ij} = \frac{1}{1 + \lVert \mathbf{c}_i - \mathbf{c}_j \rVert_2}, \qquad i \neq j
$$

对角线置零。

这相当于给图结构一个平滑偏置：

- 距离近的节点更容易连边
- 距离远的节点先验权重更小

#### 5.1.3 自适应边权

对节点对 $(i,j)$，代码构造：

$$
\mathbf{g}_{ij} = [\mathbf{h}_i, \mathbf{h}_j, |\mathbf{h}_i-\mathbf{h}_j|, \mathbf{h}_i \odot \mathbf{h}_j]
$$

再通过 `edge_mlp` 得到非负边调制项：

$$
e_{ij} = \text{ReLU}(\text{MLP}(\mathbf{g}_{ij}))
$$

于是原始邻接矩阵为：

$$
\tilde{A}_{ij} =
\begin{cases}
e_{ij} p_{ij}, & i \neq j \\
w_{\text{self}}, & i=j
\end{cases}
$$

#### 5.1.4 对称归一化

最后做标准 GCN 风格归一化：

$$
\mathbf{D}_{ii} = \sum_j \tilde{A}_{ij}
$$

$$
\mathbf{A} = \mathbf{D}^{-1/2} \tilde{\mathbf{A}} \mathbf{D}^{-1/2}
$$

这样做的作用是避免高连接度节点在聚合时数值过大。

### 5.2 时域分支

对应代码：第 182-242 行和第 383-388 行。

时域分支由 GCN 和多尺度 TCN 两部分组成。

#### 5.2.1 GCN

输入是每个时刻的变量向量，代码先做一个线性变换：

$$
\mathbf{z}_{s,n} = \mathbf{W}_g x_{s,n} + b_g
$$

因为这里 `in_dim=1, out_dim=1`，所以本质上就是一个标量仿射变换。

然后按图邻接做聚合：

$$
\mathbf{h}^{\text{gcn}}_{s} = \mathbf{A}\mathbf{z}_{s}
$$

对单个节点写开就是：

$$
h^{\text{gcn}}_{s,i} = \sum_{j=1}^{N} A_{ij} z_{s,j}
$$

这一步让每个变量在每个时刻都能吸收其它变量的信息。

#### 5.2.2 多尺度 TCN

`MultiScaleTCN` 使用 3 个深度卷积核：`3, 5, 9`。

对每个变量 $n$ 和每个卷积尺度 $k$：

$$
h^{(k)}_{s,n} = \sum_{\tau} w^{(k)}_{n,\tau} \, x_{s+\tau,n}
$$

这里因为 `groups=num_nodes`，所以每个变量独立做时间卷积，不直接在卷积层里混变量。

代码设定 `causal=False`，因此 padding 是对称的。这说明该模块不是做在线预测，而是在窗口内部利用双向上下文做重构。

三种尺度的输出堆叠后，再用一个线性层在尺度维上融合：

$$
h^{\text{tcn}}_{s,n} = \sum_{k \in \{3,5,9\}} \alpha_k h^{(k)}_{s,n}
$$

#### 5.2.3 时域分支输出

两部分直接相加：

$$
\mathbf{H}^{\text{time}} = \text{LayerNorm}(\mathbf{H}^{\text{gcn}}) + \mathbf{H}^{\text{tcn}}
$$

代码实现是先对 GCN 输出做 `LayerNorm`，再与 TCN 输出相加。

### 5.3 频域分支 `FrequencyImputer`

对应代码：第 247-301 行。

对每个变量的长度为 $S$ 的时间序列做实数 FFT：

$$
\mathbf{X}^{\mathcal{F}} = \text{RFFT}(\mathbf{X})
$$

设频域复数谱为：

$$
\mathbf{X}^{\mathcal{F}} = \mathbf{R} + i\mathbf{I}
$$

代码把实部和虚部拼接：

$$
\mathbf{f} = [\mathbf{R}, \mathbf{I}]
$$

然后走两条 MLP：

1. 频率注意力：
   $$
   \mathbf{a} = \sigma(\text{MLP}_{\text{att}}(\mathbf{f}))
   $$

2. 频谱增强：
   $$
   [\Delta \mathbf{R}, \Delta \mathbf{I}] =
   \text{MLP}_{\text{enh}}(\mathbf{f})
   $$

再把注意力作用到增强项上：

$$
\hat{\mathbf{R}} = \mathbf{a} \odot \Delta \mathbf{R}, \qquad
\hat{\mathbf{I}} = \mathbf{a} \odot \Delta \mathbf{I}
$$

以残差方式修正原频谱：

$$
\tilde{\mathbf{X}}^{\mathcal{F}}
= \mathbf{X}^{\mathcal{F}} + (\hat{\mathbf{R}} + i\hat{\mathbf{I}})
$$

最后回到时域：

$$
\mathbf{H}^{\text{freq}} = \text{IRFFT}(\tilde{\mathbf{X}}^{\mathcal{F}})
$$

随后再做 `LayerNorm`。

它的直觉是：

- 时域分支擅长局部趋势和变量关系
- 频域分支擅长周期、振荡和谱结构

### 5.4 门控融合

对应代码：第 306-333 行、第 394-399 行。

把时域和频域输出沿变量维拼接后，喂给一个 1D 卷积门控网络，得到：

$$
\mathbf{Z} = \sigma(\text{Conv1d}([\mathbf{H}^{\text{time}}, \mathbf{H}^{\text{freq}}]))
$$

其中 $\mathbf{Z}_{s,n} \in (0,1)$ 表示在时刻 $s$、变量 $n$ 上更相信哪一个分支。

融合结果：

$$
\mathbf{X}^{\text{imp}}
= \text{LayerNorm}\left(
\mathbf{Z} \odot \mathbf{H}^{\text{time}}
+
(1-\mathbf{Z}) \odot \mathbf{H}^{\text{freq}}
\right)
$$

这一步的意义是做“软选择”：

- $Z_{s,n}$ 大，更多依赖时域分支
- $Z_{s,n}$ 小，更多依赖频域分支

### 5.5 缺失值填充

模型并不是直接把融合结果作为最终输出，而是只用它替换输入中缺失的位置：

$$
\mathbf{X}^{\text{filled}}
= \mathbf{X}^{\text{in}} \odot (1-\mathbf{M}^{\text{in}})
+ \mathbf{X}^{\text{imp}} \odot \mathbf{M}^{\text{in}}
$$

这意味着：

- 观测位置保留原输入
- 原始缺失位和训练时随机遮挡位都由融合分支负责补全

### 5.6 Transformer 重构头

对应代码：第 359-413 行。

对于每个时间步 $s$，代码把变量向量映射到 `d_model=64`：

$$
\mathbf{u}_s = \mathbf{W}_{\text{in}} \mathbf{x}^{\text{filled}}_s + \mathbf{b}_{\text{in}}
$$

再加上两种位置相关信息：

1. 可学习位置编码 $\mathbf{p}_s$
2. 基于采样类型的 embedding

代码中采样类型由

$$
\text{type}(s) = s \bmod r
$$

生成，其中 `sampling_rate = 6`。于是 token 表示为：

$$
\mathbf{z}^{(0)}_s
= \mathbf{u}_s + \mathbf{p}_s + \mathbf{e}_{\text{type}(s)}
$$

随后经过两层 Transformer Encoder：

$$
\mathbf{z}^{(L)} = \text{TransformerEncoder}(\mathbf{z}^{(0)})
$$

最后投影回变量空间：

$$
\mathbf{x}^{\text{rec}}_s
= \mathbf{W}_{\text{out}} \mathbf{z}^{(L)}_s + \mathbf{b}_{\text{out}}
$$

所以最终输出是：

$$
\mathbf{X}^{\text{rec}} \in \mathbb{R}^{B \times S \times N}
$$

---

## 6. 损失函数

对应代码：`mra.py` 第 420-449 行。

总损失由三部分组成：

$$
\mathcal{L}
= \mathcal{L}_{\text{recon}}
+ \lambda_f \mathcal{L}_{\text{freq}}
+ \lambda_s \mathcal{L}_{\text{sparse}}
$$

其中代码默认：

$$
\lambda_f = 0.1, \qquad \lambda_s = 0.1
$$

### 6.1 时域重构损失

只在 `target_mask` 指定的位置上监督：

$$
\mathcal{L}_{\text{recon}}
=
\frac{
\left\|
(\mathbf{X}^{\text{rec}} - \mathbf{X}) \odot \mathbf{T}
\right\|_F^2
}{
\sum \mathbf{T}
}
$$

这里 $\mathbf{T}$ 就是前面说的 `target_mask`。

关键点：

- 默认情况下只监督“训练时额外随机遮挡掉的观测位置”
- 这使它成为一个 masked denoising / masked reconstruction 任务
- 原始缺失位并不直接参与这个重构监督

### 6.2 频域一致性损失

代码先构造：

$$
\mathbf{X}^{\text{freq-target}}
=
\text{stopgrad}(\mathbf{X}^{\text{rec}}) \odot (1-\mathbf{T})
+ \mathbf{X} \odot \mathbf{T}
$$

这意味着：

- 在目标位置，用真实值 $\mathbf{X}$ 作为频域监督
- 在非目标位置，用 `detach` 后的当前预测值占位

数值上，相减后有：

$$
\mathbf{X}^{\text{rec}} - \mathbf{X}^{\text{freq-target}}
=
(\mathbf{X}^{\text{rec}} - \mathbf{X}) \odot \mathbf{T}
$$

也就是说，频域损失虽然在频谱空间计算，但本质上仍只由目标位置上的误差驱动。

对有效样本（原始观测比例大于 50%）：

$$
\mathcal{L}_{\text{freq}}
=
\left\|
\left|
\mathcal{F}(\mathbf{X}^{\text{rec}})
\right|
-
\left|
\mathcal{F}(\mathbf{X}^{\text{freq-target}})
\right|
\right\|_2^2
$$

代码只比较幅值谱，不比较相位谱。

它的作用是：

- 不只要求时间点逐点接近
- 还要求整体频谱结构接近

### 6.3 图熵正则

代码中的第三项是：

$$
\mathcal{L}_{\text{sparse}}
= -\frac{1}{B} \sum_{b=1}^{B} \frac{1}{N}
\sum_{i=1}^{N}\sum_{j=1}^{N}
A^{(b)}_{ij} \log(A^{(b)}_{ij} + 10^{-8})
$$

这其实更接近“负熵”而不是传统的 L1 稀疏项。

因为熵

$$
H(\mathbf{a}_i) = -\sum_j a_{ij}\log a_{ij}
$$

当分布更尖锐、更集中时，熵更小。最小化这里的负号版本，会鼓励每一行邻接权重更集中，而不是均匀摊开。

所以它的作用可以理解为：

- 抑制“平均连所有点”的模糊图
- 鼓励更有选择性的依赖关系

---

## 7. 参数更新

对应代码：`mra.py` 第 538-579 行。

每个 seed 的训练步骤是：

1. 初始化模型
2. 用 Adam 更新参数
3. 对每个 batch 计算总损失
4. 反向传播
5. 做梯度裁剪
6. 以 epoch 平均 loss 最小的参数作为该 seed 的最终模型

数学上，Adam 更新可以抽象写成：

$$
\theta \leftarrow \text{Adam}\left(\theta, \nabla_\theta \mathcal{L}\right)
$$

代码中的梯度裁剪：

$$
\lVert \nabla_\theta \mathcal{L} \rVert_2 \le 1
$$

对应：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

它的主要作用是稳定训练，防止梯度爆炸。

另外，`train()` 不是只训练一个模型，而是训练 3 个 seed 的 ensemble：

$$
\theta^{(40)}, \theta^{(41)}, \theta^{(42)}
$$

如果对应 checkpoint 已存在，则直接加载，不再重新优化。

---

## 8. 训练后如何变成异常分数

对应代码：`mra.py` 第 456-510 行、第 689-759 行。

### 8.1 单模型异常分数

对一个窗口样本，模型输出重构值 $\mathbf{X}^{\text{rec}}$。代码只在原始观测位置上计算均方误差：

$$
\text{score}
=
\frac{
\left\|
(\mathbf{X}^{\text{rec}} - \mathbf{X}) \odot (1-\mathbf{M})
\right\|_F^2
}{
\sum (1-\mathbf{M})
}
$$

这样可以避免把“本来就缺失的位”计入异常分数。

### 8.2 多 seed 集成

对 3 个 seed，分别得到分数序列：

$$
\mathbf{s}^{(40)}, \mathbf{s}^{(41)}, \mathbf{s}^{(42)}
$$

再做均值集成：

$$
\bar{\mathbf{s}}
= \frac{1}{3}
\left(
\mathbf{s}^{(40)} + \mathbf{s}^{(41)} + \mathbf{s}^{(42)}
\right)
$$

其直觉是降低单次随机初始化带来的方差。

### 8.3 原始阈值

先从训练集分数估计阈值：

$$
\tau_{\text{raw}} = \mu_{\text{train}} + \sigma_{\text{train}}
$$

如果测试分数大于这个阈值，就认为当前窗口更像异常。

### 8.4 EWMA 平滑

代码进一步对分数做指数滑动平均：

$$
\tilde{s}_1 = s_1
$$

$$
\tilde{s}_t = \alpha s_t + (1-\alpha)\tilde{s}_{t-1}, \qquad \alpha = 0.02
$$

它相当于一个低通滤波器，能减少单点抖动。

### 8.5 持续故障判定

平滑后再根据训练集平滑分数设置阈值：

$$
\tau = \mu(\tilde{\mathbf{s}}_{\text{train}})
+ 1.5 \cdot \sigma(\tilde{\mathbf{s}}_{\text{train}})
$$

点报警为：

$$
\hat{y}_t^{\text{point}} =
\mathbb{I}[\tilde{s}_t > \tau]
$$

但代码并不直接采用点报警，而是要求异常至少持续 `150` 个窗口：

$$
\hat{y}_t^{\text{persistent}} = 1
\quad \text{only if a consecutive run of ones has length} \ge 150
$$

这一步是经验上的后处理，目的是减少瞬时误报，更符合“持续故障”场景。

---

## 9. 代码实现上的几个关键结论

这些不是额外假设，而是直接来自当前代码行为。

### 9.1 训练本质是“随机遮挡重构”，不是普通自编码

真正参与监督的主要是：

- 原本观测到
- 但训练时被随机隐藏

的位置。它更接近 masked modeling。

### 9.2 原始缺失值不会直接进入重构误差监督

因为：

- `target_mask` 只来自随机遮挡或全观测兜底
- 异常分数也只在原始观测位计算

所以模型不会因为“原始 NaN 没法对齐真实值”而被直接惩罚。

### 9.3 图正则并不是标准 L1 稀疏，而是低熵约束

它鼓励的是“连接分布更集中”，不是简单地把所有边都压小。

### 9.4 `train()` 里其实包含了训练后评估与检测

函数名虽然叫 `train()`，但它实际做了三件事：

1. 训练或加载 ensemble
2. 计算 train/test 分数
3. 输出最终检测指标和图像

所以它更像“完整实验入口”，而不是纯训练入口。

### 9.5 测试标签是代码里人为规定的

`build_test_labels()` 直接把测试集后半段标成异常：

$$
y_t =
\begin{cases}
0, & t < T/2 \\
1, & t \ge T/2
\end{cases}
$$

这说明当前评估协议依赖于数据文件的组织方式，而不是从文件里读取外部标签。

---

## 10. 一句话总结

当前 `mra.py` 的训练流程可以概括为：

> 先把多变量缺失时间序列切成窗口，在窗口内随机再遮掉一部分观测值；然后用“自适应图学习 + 时域 GCN/TCN + 频域 FFT 修复 + 门控融合 + Transformer 重构头”去恢复这些被遮挡的位置；最后用重构误差作为异常分数，并通过集成、EWMA 和持续段过滤得到故障判定。

