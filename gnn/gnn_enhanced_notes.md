# 📘 GNN 系统增强版笔记（最终结构树版 · Markdown 内容）

- [📘 GNN 系统增强版笔记（最终结构树版 · Markdown 内容）](#-gnn-系统增强版笔记最终结构树版--markdown-内容)
- [A. Graph Construction（图构建阶段）](#a-graph-construction图构建阶段)
  - [A0. Why we use Graph?](#a0-why-we-use-graph)
  - [A1. Graph Structure（图的结构定义）](#a1-graph-structure图的结构定义)
  - [A2. Node Feature（节点特征）](#a2-node-feature节点特征)
    - [weight（重量）](#weight重量)
    - [value（价值）](#value价值)
    - [ratio（密度）](#ratio密度)
    - [one-hot（独热编码）](#one-hot独热编码)
    - [其他统计特征](#其他统计特征)
  - [A3. Edge Construction（边的构建方式）](#a3-edge-construction边的构建方式)
    - [0. Edge（边）= 节点之间的联系](#0-edge边-节点之间的联系)
    - [1. Fully-connected（全连接）](#1-fully-connected全连接)
    - [2. kNN（k 最近邻图）解释](#2-knnk-最近邻图解释)
    - [3. 相似度阈值图（similarity \> τ 才连边）](#3-相似度阈值图similarity--τ-才连边)
  - [A4. Edge Weight（可选）](#a4-edge-weight可选)
  - [A5. Self-loop（可选）](#a5-self-loop可选)
- [B. Node Embedding（节点嵌入表示）](#b-node-embedding节点嵌入表示)
  - [B1. What is Embedding](#b1-what-is-embedding)
  - [B2. 初始 Embedding](#b2-初始-embedding)
- [C. Message Passing（核心机制）](#c-message-passing核心机制)
  - [C0. Overall Intro](#c0-overall-intro)
  - [C1. Neighbor Collection](#c1-neighbor-collection)
  - [C2. Message Function](#c2-message-function)
  - [C3. Aggregation（聚合）](#c3-aggregation聚合)
  - [C4. Update（更新阶段）：用汇总信息更新节点 embedding](#c4-update更新阶段用汇总信息更新节点-embedding)
  - [C5. Multi-layer Message Passing（多层 GNN）让信息看得更远](#c5-multi-layer-message-passing多层-gnn让信息看得更远)
  - [C6. 从“特征向量”到“选择概率”：MLP 的决策作用](#c6-从特征向量到选择概率mlp-的决策作用)
  - [C7. Model Evaluation（与 DP 最优解比对）](#c7-model-evaluation与-dp-最优解比对)
  - [C8. 最终结构检查（逻辑链）](#c8-最终结构检查逻辑链)
- [D. Readout（图级别聚合 · 可选）](#d-readout图级别聚合--可选)
- [E. Decision Making（决策阶段）](#e-decision-making决策阶段)
  - [E1. MLP Classifier（将 embedding 转换成选择概率）](#e1-mlp-classifier将-embedding-转换成选择概率)
  - [E2. Thresholding（把概率转换成具体决策）](#e2-thresholding把概率转换成具体决策)
  - [Why 0.5?（为什么选 0.5？）](#why-05为什么选-05)
  - [E3. Decision Example（完整例子）](#e3-decision-example完整例子)
  - [E4. 决策阶段的本质总结（给小白）](#e4-决策阶段的本质总结给小白)
- [F. Training Pipeline（训练流程）](#f-training-pipeline训练流程)
  - [F1. Loss Function（损失函数）](#f1-loss-function损失函数)
    - [1) 分类损失（Binary Cross Entropy, BCE）](#1-分类损失binary-cross-entropy-bce)
    - [2) 容量约束惩罚（Knapsack 关键）](#2-容量约束惩罚knapsack-关键)
    - [3) 总损失（综合分类＋约束）](#3-总损失综合分类约束)
      - [（可选补充：更高级的损失设计）](#可选补充更高级的损失设计)
  - [F2. Optimizer（优化器）](#f2-optimizer优化器)
    - [● Adam（首选）](#-adam首选)
    - [● 可选技巧](#-可选技巧)
  - [F3. Batch 构建方式（Graph-Level Mini-Batch）](#f3-batch-构建方式graph-level-mini-batch)
  - [F4. Epoch Training Loop（完整训练循环）](#f4-epoch-training-loop完整训练循环)
    - [**Step 1. Forward（前向传播）**](#step-1-forward前向传播)
    - [**Step 2. Compute Loss（计算损失）**](#step-2-compute-loss计算损失)
    - [**Step 3. Backward（反向传播）**](#step-3-backward反向传播)
    - [**Step 4. Optimizer Step（更新参数）**](#step-4-optimizer-step更新参数)
  - [F5. Model Monitoring（训练过程监控）](#f5-model-monitoring训练过程监控)
    - [1. Train / Val Loss（训练 / 验证损失）](#1-train--val-loss训练--验证损失)
    - [2. Penalty（超重惩罚项）](#2-penalty超重惩罚项)
    - [3. Accuracy（预测与 DP 最优解的匹配度）](#3-accuracy预测与-dp-最优解的匹配度)
    - [4. 可解释性检查（Interpretability）](#4-可解释性检查interpretability)
  - [F6. 训练流程总结（最重要的五句话）](#f6-训练流程总结最重要的五句话)
- [📌 GNN 全流程总结](#-gnn-全流程总结)

---
# A. Graph Construction（图构建阶段）

## A0. Why we use Graph?
在机器学习里，数据通常有结构：

- **图像**：像素组成网格
- **句子**：词组成序列  
- **表格**：属性组成行

但有些数据天然是"点＋关系"：
- 物品之间的相似性 → Knapsack
- 社交网络：人之间的关系
- 推荐系统：商品之间的共同购买关系
- 分子化学：原子之间怎么连接
- 地图：城市之间路线

这些都不是数组，而是**图结构（Graph）**：
- **节点（Node）** = "一个对象"（例如一个物品）
- **边（Edge）** = "两个对象之间的关系"（例如相似度高）

GNN 专门处理这种数据。

## A1. Graph Structure（图的结构定义）
在实现任何 GNN 之前必须明确图的结构：
- **图类型：** 无向图 / 有向图（knapsack 多为无向图）
- **自环（Self-loop）：** 是否让节点与自己连边
- **加权边（Weighted edges）：** 是否使用相似度作为边的权重
  - 例如 cosine similarity / ratio 差值 / weight 差值

这些定义决定 message passing 的数学形式。

---
## A2. Node Feature（节点特征）
### weight（重量）
### value（价值）
### ratio（密度）
Ratio（价值密度）= value / weight

**定义：** 每 1 单位重量能换多少钱？

**示例表格：**
| 重量 | 价值 | ratio |
|------|------|-------|
| 3    | 9    | 3     |
| 3    | 4    | 1.33  |

**应用价值：**
- ratio 越高，越"划算"
- GNN 可以用 ratio 作为特征，让它学会哪些物品值得优先选
### one-hot（独热编码）
**作用：** 告诉模型"这是第几个物品"或"是什么类别"。

**示例：**
假设你有 5 个物品，物品 3 的 one-hot 就是：[0, 0, 1, 0, 0]

**特点：**
- 只亮一个灯，表示它的身份
- GNN 用它来区分不同物品
### 其他统计特征
➡ 所有特征会作为 embedding 的输入

---
## A3. Edge Construction（边的构建方式）
### 0. Edge（边）= 节点之间的联系

**边的连接方式：**
- `ratio` 相似度高 → 连边
- `kNN`（k 最近邻） → 连最相似的 k 个物品
- `fully-connected`（全连接） → 任意两个物品都连边

**为什么需要边？**
因为 message passing 需要知道"向谁传消息"。

常见方式：
### 1. Fully-connected（全连接）
### 2. kNN（k 最近邻图）解释
**kNN 图** = 每个节点连接与自己最相似的 k 个节点,把每个物品和"最像它的一些物品"连起来

**连接规则：**
- A 和 B 的 ratio 很接近 → 他们连边
- A 和 C 的重量很接近 → 他们连边

**作用：**
GNN 在"消息传递"时就知道："A"应该从"像它的物品"那里学经验

**优势：**
GNN 只在"相似物品"之间传递信息，效率高且有意义。
### 3. 相似度阈值图（similarity > τ 才连边）


---
## A4. Edge Weight（可选）
- 用 ratio 差值或 cosine similarity 作为边权
- 影响邻居权重和消息增益

---
## A5. Self-loop（可选）
- 让节点的历史信息也参与更新

---
# B. Node Embedding（节点嵌入表示）

## B1. What is Embedding
要理解 GNN，一定要先理解 embedding（嵌入）。

**⭐ Embedding** = "把一个东西变成一个向量（vector）"

- 一个词：变成 300 维 vector（word2vec）
- 一个物品：变成 64 维 vector  
- 一个节点：变成 128 维 vector

**本质意义**：让模型能用数学方式处理复杂对象（词、物体、节点）。因为向量可以捕捉"语义关系"和"结构关系"。
**举例**：
- "苹果"和"香蕉" → embedding 很像
- "价值高、重量低"的物品之间 → embedding 很像
- "有相似边和邻居的节点" → embedding 很像

## B2. 初始 Embedding
$$
h_i^(0) = Linear(x_i)
$$
- 将原始特征映射到隐藏空间

---
# C. Message Passing（核心机制）
## C0. Overall Intro
>Message Passing 是 GNN 的核心思想：节点与邻居交换信息，通过多轮迭代不断强化自身表示。类比为“人类交流 → 成长”。

>其基本流程为：
1.识别邻居节点（基于图结构）
2.从邻居中提取关键信息（Message）
3.聚合所有邻居信息（Aggregation）
4.使用聚合后的信息更新自身 embedding（Update）

**具体工作流程：**
1. 以物品 A 为例，它会先识别与自己相连的邻居（比如价值密度 ratio 相似、存在组合关联的其他物品）；
2. 从这些邻居那里“学习”关键信息（比如邻居的重量、价值、是否适合一起被选中等）；
3. 把收集到的邻居信息与自身信息整合，最终更新自己的特征向量（embedding）。

**数学表达（不用背，理解逻辑即可）：**
$$new\_state(A) = f(A, \sum(\text{neighbors' states}))$$
- 公式中，`A` 代表物品 A 的自身原始状态，`sum(neighbors' states)` 就是对邻居信息的“合并操作”，`f` 是整合自身与邻居信息的更新函数。

**更标准化的 GNN 形式：**
 $$ h_i^{(l+1)} = UPDATE(h_i^{(l)}, AGGREGATE({ MESSAGE(h_j) | j ∈ N(i) }))$$


**核心要点:**
- 随着多次 Message Passing 的迭代，GNN 会逐渐学会“哪些物品之间组合更可能形成高价值、不超重的集合”；
- 这种能自动学习“组合关联”的能力，正是 GNN 天然适合解决背包问题的关键。

**（建议补充：Message Passing 与传统 MLP/CNN 的本质区别）**


---
## C1. Neighbor Collection
邻居由 **图结构阶段（Graph Construction）** 决定。例如：
- 基于 ratio 相似度连接
- 基于权重/价值关系连接
- 基于组合模式连接

节点 i 的邻居集合记为：**N(i)**。

**（建议补充：邻接矩阵/邻接列表在实际实现中的作用）**

---
## C2. Message Function

从邻居节点中抽取“要发送的信息”。常见形式：
- **MLP(h_j)**：对邻居特征做一次线性变换并加激活
- 若图有边特征，可使用 MESSAGE(h_i, h_j, e_{ij})（你目前暂不需要）


**（建议补充：为什么 Message Function 可以学习出“组合关系”）**

---
## C3. Aggregation（聚合）
Aggregation（聚合）：Message Passing 中的核心子步骤

上面公式里的 `sum(neighbors' states)`，本质就是 Aggregation——它是 Message Passing 流程中“合并邻居消息”的关键环节，没有聚合，分散的邻居信息无法被节点有效利用。

**为什么必须聚合？**
一个节点可能连接多个邻居（比如物品 A 连了 10 个其他物品），如果把每个邻居的消息都单独处理，不仅效率低，还会导致信息冗余混乱，因此需要通过聚合将分散的邻居消息合并为统一的“综合信息”。

**常见聚合方式：**
- `sum`（求和）：简单粗暴，直接累加所有邻居的消息，保留信息总量；
- `mean`（平均）：对邻居消息取平均，平滑极端值的影响；
- `attention`（注意力加权）：给更重要的邻居（比如与当前物品组合价值更高的邻居）分配更高权重，让关键信息发挥更大作用。

**Aggregation 的输出作为下一步 Update 的输入。**
**（建议补充：不同 Aggregation 对模型表达能力的影响）**

---
## C4. Update（更新阶段）：用汇总信息更新节点 embedding
更新公式：
$$
h_i^{(1)} = f(h_i^{(0)}, sum(Aggregate(h_j^{(0)})))
$$

- \( h_i^{(0)} \)：物品 i 一开始的特征（比如 `[w, v, ratio,…]`）→ **Update 阶段的“自身信息”**
- \( \sum_{j \in N(i)} \text{Aggregate}(h_j^{(0)}) \)：邻居特征的聚合结果 → **Aggregate 阶段的输出**
- \( h_i^{(1)} \)：更新后的“更聪明”的特征，已经包含了其他物品的信息 → **Update 阶段的最终结果**
类比：与朋友交流后“更懂得如何做选择”。


**（建议补充：Update 与残差连接/归一化的关系）**



---
## C5. Multi-layer Message Passing（多层 GNN）让信息看得更远

多层 GNN 的核心逻辑是“多次迭代消息传递”——每一层对应一次节点间的信息交流，随着层数增加，节点能“看到”的邻居范围会不断扩大，最终捕捉到整张图的全局关联信息。

**层级与信息传播范围的关系：**
- 每一层 = 一次完整的 Message Passing（信息交流）；
- 多层 = 让节点突破直接邻居的限制，看到更远距离的关联节点；
- 具体传播范围：
  - 第 1 层：仅能获取**直接邻居**的信息（局部关联）；
  - 第 2 层：能获取**邻居的邻居**的信息（次全局关联）；
  - 第 3 层及以上：可逐步覆盖整张图的信息（全局关联）。

**多层 GNN 的递进效果**
多堆几层 GNN：
- 第一层：学“局部”的关系（仅直接邻居）
- 第二层：学“更全局”的关系（邻居的邻居）

最后我们得到每个物品一个 embedding，比如：
>h1 = [0.8, 1.2, -0.5, ...] # 整合了邻居信息的物品 1 表示
h2 = [1.1, 0.3, 0.7, ...] # 整合了邻居信息的物品 2 表示
h3 = [-0.2, 2.0, 0.1, ...] # 整合了邻居信息的物品 3 表示


这些就是 “考虑了其他物品后的表示”。


## C6. 从“特征向量”到“选择概率”：MLP 的决策作用
经过多层消息传递后，每个节点（物品）会得到一个整合了全局信息的 embedding \( h_i \)。为了将这个高维特征向量转化为“是否选择该物品”的明确决策，我们会给每个 \( h_i \) 搭配一个小型前馈网络（MLP），通过线性变换+激活函数输出 0~1 之间的概率：

\[ p_i = \sigma( W \cdot h_i + b ) \]
- \( W \) 和 \( b \)：MLP 的可学习参数（权重和偏置）；
- \( \sigma \)：激活函数（如 Sigmoid），作用是将输出压缩到 0~1 区间，得到概率值；
- \( p_i \)：物品 \( i \) 被选中的概率。

**直觉理解：概率的实际含义**
- \( p_i \) 越接近 1 → 模型判断“该物品被选中能提升总价值，应该选”；
- \( p_i \) 越接近 0 → 模型判断“该物品选中后可能超重或价值较低，不应该选”。

**最终决策：概率阈值化**
得到所有物品的选择概率后，我们通过“阈值化”将概率转化为明确的 0/1 决策（对应背包问题的“拿”或“不拿”）：
\[ 
x_i = 
\begin{cases} 
1 & \text{if } p_i > 0.5 \\
0 & \text{if } p_i \leq 0.5 
\end{cases}
\]

**示例：从概率到最终答案**
假设经过多层 GNN 计算后，得到 3 个物品的选择概率：\( p_1=0.92 \)、\( p_2=0.87 \)、\( p_3=0.31 \)，阈值设为 0.5，则最终决策为：
\[ \text{预测的 } x = [1, 1, 0] \]

## C7. Model Evaluation（与 DP 最优解比对）
将这个预测结果与动态规划（DP）等方法算出的最优答案（如同样是 [1,1,0]）进行比对：
- 若预测结果与最优答案高度一致 → 说明模型已学会背包问题的组合优化规律，学得不错；
- 若偏差大 → 调整以下内容：
- GNN 层数
- MLP 结构
- 邻接构造策略
- Loss 设计


**（建议补充：常用的训练损失，例如二分类交叉熵）**

## C8. 最终结构检查（逻辑链）
下面是本笔记的逻辑链条，确保 Message Passing 讲解完整、自洽：


1. **整体流程（C0）**：Message Passing 的大图景
2. **邻居来源（C1）**：信息来自哪里
3. **信息从邻居抽取（C2）**
4. **将多邻居信息聚合（C3）**
5. **更新节点自身表示（C4）**
6. **多层传播构建全局理解（C5）**
7. **将 embedding 转换成 0/1 决策（C6）**
8. **用 DP 对比评估（C7）**

---
# D. Readout（图级别聚合 · 可选）
如果任务是图分类（graph-level），需要：
- sum pooling
- mean pooling
- max pooling
- attention pooling

Knapsack 多为 node-level，但 Readout 是 GNN 基础组件。

---
# E. Decision Making（决策阶段）

Message Passing（消息传递）结束后，每个物品都会得到一个高维的 embedding（特征向量）。  
决策阶段的任务是：**把这些 embedding 转换成“选 or 不选”的最终判断（0/1）**。

这一阶段包含两个核心部分：

1. 用 MLP 预测每个物品被选中的概率  
2. 通过阈值把概率转成明确的 0/1 决策  

---

## E1. MLP Classifier（将 embedding 转换成选择概率）

经过多层 GNN 后，每个物品节点 i 有一个 embedding：

$$h_i = [0.8, 1.2, -0.5, ...]$$

但 embedding 不能直接告诉我们“要不要选”。  
因此我们需要一个 **多层感知机（MLP）** 对每个 h_i 进行分类：

$$p_i = σ(W · h_i + b)$$

解释如下：

- **W、b**：可学习参数（矩阵和偏置）
- **σ**：Sigmoid 函数，把输入压缩到 0～1 之间
- **p_i**：模型认为物品 i 被选中的概率

直觉理解：

- p_i = 0.93 → 模型非常认为应该选  
- p_i = 0.12 → 模型认为不能选  
- p_i 越接近 1，越“值得放入背包”  

MLP 的作用相当于一个“评分器（scoring function）”，根据 embedding 判断物品是否合适。

---

## E2. Thresholding（把概率转换成具体决策）

MLP 输出的是概率，但背包问题要求输出 **0 或 1**。

常用方法：设定阈值 0.5  
$$x_i = 1 if p_i > 0.5  else 0$$


说明：

- 若 p_i > 0.5 → 选取该物品  
- 若 p_i ≤ 0.5 → 不选取  

这是最常见、最简单的二分类决策方式。

---

## Why 0.5?（为什么选 0.5？）

Sigmoid 输出区间是 0～1：  
- 0.5 是“模型不确定”的中点  
- >0.5 表示偏向于“选”  
- <0.5 表示偏向于“不选”

当然，你也可以调整阈值（例如 0.6 或 0.4），以改变模型的：

- **探索性**（更容易选物品）  
- **保守性**（更不容易选物品）  

在实际项目中，有时还会动态调节阈值来控制是否超重。

---

## E3. Decision Example（完整例子）

假设 3 个物品的概率是：

| 物品 | p_i | 决策 x_i |
|------|------|-----------|
| 1 | 0.92 | 1 |
| 2 | 0.73 | 1 |
| 3 | 0.31 | 0 |

最终输出：
x = [1, 1, 0]

含义：选第 1、2 件物品，不选第 3 件。

---

## E4. 决策阶段的本质总结（给小白）

1. **GNN 已经学好 embedding，但不能直接告诉你该选谁**  
2. **MLP 负责把 embedding 转成“概率”**  
3. **阈值把概率转成明确的 0 or 1**  
4. **最终得到的 x 就是背包的预测解**

决策阶段是 GNN 推理流程的最后一步，把“理解”变成“动作”。


---
# F. Training Pipeline（训练流程）

GNN 训练的目标是：  
让模型从大量 Knapsack 实例中学习 **如何选择物品**，同时满足 **价值最大化与容量约束**。

本节围绕：  
**损失函数 → 优化器 → Batch → Epoch → 模型监控**  
构建完整的训练流程知识体系。

---

## F1. Loss Function（损失函数）

在 Knapsack 中，损失需要同时处理两件事：  
1. **分类准确性**：物品是否应该被选中（与 DP 标签对齐）  
2. **容量限制**：总重量不能超过 W

---

### 1) 分类损失（Binary Cross Entropy, BCE）
用于学习每个物品的 0/1 选择模式：

$$
loss_{cls} = BCE(p_i, y_i)
$$

- \( p_i \)：模型预测物品 i 被选中的概率  
- \( y_i \)：DP 解中的最优标签（0/1）

---

### 2) 容量约束惩罚（Knapsack 关键）
使模型“意识到”超重是不允许的：

$$
penalty = \lambda \cdot \max(0,\ \sum(w_i \cdot x_i) - W)
$$

解释：
- 总选中物品的重量若超过容量 W → 产生惩罚  
- \( \lambda \) 控制惩罚力度（常用范围 1～10）

优点：
- 容易实现  
- 允许模型在训练早期“试错”，逐渐学会约束

---

### 3) 总损失（综合分类＋约束）
最终的训练目标：

$$
loss = loss_{cls} + penalty
$$

---

#### （可选补充：更高级的损失设计）
你后续可以加入（非必需）：
- **Soft Knapsack Constraint**（让容量约束更平滑）
- **Lagrangian Relaxation**（强约束）
- **价值导向奖励项**（鼓励高价值组合）

---

## F2. Optimizer（优化器）

训练 GNN 时的典型优化器与配置：

### ● Adam（首选）
- 收敛稳定、对超参数不敏感
- 常用学习率 lr：`1e-3 ~ 5e-4`

### ● 可选技巧
- **学习率衰减 scheduler**（训练后期自动变慢）
- **weight decay**（提升泛化能力）
- **gradient clipping**（避免梯度爆炸）

（建议补充：为何 SGD 不适用 GNN → 因为图结构梯度噪声大）

---

## F3. Batch 构建方式（Graph-Level Mini-Batch）

在 Knapsack 中：

- 每个 knapsack 实例 = 1 张图  
- 1 个 batch = 多张图拼成的“大图 batch”

使用框架：
- PyTorch Geometric（自动帮你合并图、生成 batch mask）

批处理后的优势：
- **训练稳定**
- **显著加速**
- **提升泛化能力**

Batch 尺寸典型值：`16 ~ 64` 图 / batch

---

## F4. Epoch Training Loop（完整训练循环）

一个标准的 GNN 训练 epoch 包含以下步骤：

---

### **Step 1. Forward（前向传播）**
- 运行 Message Passing  
- 更新节点 embedding  
- MLP 输出每个物品的选择概率 \( p_i \)

---

### **Step 2. Compute Loss（计算损失）**
包含：
- BCE 分类损失  
- 超重惩罚项 penalty  
- 组合为最终 loss

---

### **Step 3. Backward（反向传播）**
- 自动求导计算梯度  
- 梯度会沿 Message Passing 路径反向流动

---

### **Step 4. Optimizer Step（更新参数）**
执行：

python
optimizer.step()
optimizer.zero_grad()

## F5. Model Monitoring（训练过程监控）

在训练 GNN 解决背包问题的过程中，需要监控多个关键指标，用于判断模型是否真正学到了“组合规律”与“容量约束”。

---

### 1. Train / Val Loss（训练 / 验证损失）
- 用于判断模型是否过拟合或欠拟合  
- 正常现象：  
  - Train / Val Loss 随 epoch 缓慢下降  
  - 二者差距不应过大

---

### 2. Penalty（超重惩罚项）
- 若 penalty 多个 epoch 都为 **正值** → 模型仍在超重  
- 若 penalty 趋近 0 → 模型已经学会遵守容量约束  
- 若 penalty 长期不下降，说明可能存在：  
  - 图结构构造不合理  
  - λ 惩罚系数设置过低  
  - 模型 expressiveness 不足

---

### 3. Accuracy（预测与 DP 最优解的匹配度）
- 对比预测的选取向量 x 与 DP 最优 x\*  
- Accuracy 越高，代表模型越能模仿最优策略  
- 常见评估方式：  
  - item-level accuracy  
  - knapsack-level exact match（更严格）

---

### 4. 可解释性检查（Interpretability）
- 观察节点 embedding 是否呈现合理结构  
- 常见现象：  
  - 高价值密度物品 embedding 更接近  
  - 不适合一起选的物品 embedding 会被分开  
- 可视化工具：t-SNE, PCA

---

## F6. 训练流程总结（最重要的五句话）

1. **BCE 让模型学会选对物品（模仿 DP 标签）。**
2. **Penalty 让模型学会不超重。**
3. **Batch 训练让学习过程更稳定、高效。**
4. **多次 Epoch 迭代使模型逐渐逼近 DP 的决策模式。**
5. **最终 GNN 学到的不是单个物品特性，而是“组合关系”。**


---

# 📌 GNN 全流程总结
```
图构建（node + edge + 结构定义）
        ↓
初始 embedding
        ↓
多层 message passing
        ↓
决策（MLP + threshold）
        ↓
Loss（分类 + 超重惩罚）
        ↓
训练（Adam + epoch）
```

