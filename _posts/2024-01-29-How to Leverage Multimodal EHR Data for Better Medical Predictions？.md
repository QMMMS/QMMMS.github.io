---
title: How to Leverage Multimodal EHR Data for Better Medical Predictions?
date: 2024-01-29 18:21:00 +0800

media_subpath: "/assets/img/posts/2024-01-29-How to Leverage Multimodal EHR Data for Better Medical Predictions？"
categories: [深度学习]
tags: [读论文,医疗任务]
math: true
---

> 读论文时间！
>
> [官方代码](https://github.com/emnlp-mimic/mimic)
{: .prompt-info }

## 介绍

深度学习提供了改善医疗服务质量的巨大机会。然而，电子健康记录（EHR）数据的复杂性是深度学习应用的一个挑战。

数据可以分为三种模式：

![](icu_d.png)

- 非时间相关的数据，如患者的年龄、性别等。通常，在住院期间不会改变。
- 时间序列数据，如生命体征和实验室检查结果。这些数据共享相同的特点，即它们会随着时间而变化，并且在时间上的分布是非均匀的。例如，心率和血压等生命体征会在数小时或数天内连续记录，而血液检测等实验室检查是在住院期间某个特定时间发生的离散事件。
- 临床笔记是非结构化的自由文本，通常比时间序列数据更稀疏。更重要的是，这些注释充满了缩写词、行话和不寻常的语法结构，对于非专业人士来说很难阅读和理解。

一些针对EHR数据的预处理框架，忽略了病历笔记。因此，本文的贡献可以总结如下：

- 我们首先提出了一种联合建模的方法，以区分差异：
- 从预处理管道中提取的实体数据源和临床笔记，以提高医学预测。
- 我们提出了一种融合方法，使用大型预训练模型集成时间不变数据、时序数据和临床笔记。

我们在两个任务上进行实验：诊断预测和急性呼吸衰竭 (ARF) 预测，并对融合策略进行全面探索。结果表明，我们的模型优于没有临床笔记的模型，这说明了临床笔记的重要性以及我们融合方法的有效性。

## 相关工作

为了克服特定数据集的局限性，出现对 EHR 数据提出了一个名为 FIDDLE 的系统预处理技术。本文使用了 FIDDLE 提取的数据。

由于常见的预训练语言模型 BERT 并未考虑病历的特定复杂性，因此本文使用针对病历进行预训练的 ClinicalBERT 来处理病历数据。

在这项工作中，我们受到多模式自适应门（MAG）的启发，使用注意力门融合上述三种模式。背后的核心思想是通过从其他模式派生的偏移量来调整一个模式的表示。我们认为 text/note应该是主模式。

## 模型

![](model2.png)

### 定义

根据提取数据的方法，数据分为两类。从预处理管道框架中提取的数据被划分为不变数据和时间序列数据。

给定批大小为B，时间无关特征的维度是D1，时间无关数据可以表示为：

$$
I_{ti}\in \mathcal{R}^{B\times D_1}
$$

给定按小时计数的ICU停留时间长度L，时间序列数据的维数D2，时间序列数据表示为：

$$
I_{ts}\in \mathcal{R}^{B\times L\times D_2}
$$

对于临床记录，采用分词对其进行标记化并转换为标记,给定记录的长度D3，标记 ID 可以表示为：

$$
I_{nt}\in \mathcal{R}^{B\times D_3}
$$

### 编码

我们为每种模态使用不同的编码器：

**时间无关编码**：由于时间不变数据包含患者简单而固定的信息，如年龄、性别和种族，我们认为具有ReLU激活的全连接网络足以对这些信息进行编码，给定编码特征的维度D1'，$E_{ti} \in \mathcal{R}^{B\times D_1'}$：

$$
E_{ti}=\text{ReLU}(\text{Linear}(I_{ti}))
$$

**时间序列编码**：给定时间序列数据包括每小时特征重要性。为了对这些信息进行编码，我们更喜欢具有处理时间序列的能力的模型。在本工作中，我们使用了四种不同的编码器，每种都与稍后介绍的临床 BERT 集成以创建一个基线模型。

根据不同的建模功能和它们被提出的时机，编码器可以分为两组。

1. 第一组包含长短期记忆 (LSTM) 和卷积神经网络 (CNN)。
2. 第二组包含 Star-Transfomer 和 Transfomer 编码器，因为它可以联合地根据左右上下文来学习表示，而Star-Transfomer减少了Transfomer的复杂性，同时保留了捕获局部组合和长期依赖的能力。

给定 L2'是隐藏大小，D2'是神经元的数量，ENC表示编码器，对应于上述四个编码器：LSTM、CNN、transformer 编码器和 Star-Transformer。于是 $E_{ts}' \in \mathcal{R}^{B\times L_2'}, E_{ts} \in \mathcal{R}^{B\times D_2'}$

$$
E_{ts}'=\text{ENC}(I_{ts})
$$

$$
E_{ts}=\text{ReLU}(\text{Linear}(E_{ts}'))
$$

![](ts.png)

注意：

- transformer 编码器的计算与方程略有不同，在此，E'ts 不会传递给后续的线性层，而是直接用作编码表示。
- 如果 LSTM 有多个层，则始终使用最后一个层的隐藏状态。

**临床记录编码**：我们使用预训练的 ClinicalBERT 编码临床记录。在为特定任务进行训练时，整个预训练模型将与其他编码器一起微调以实现更好的适应性。用 $E_{nt}∈\mathcal{R}^{B×D_3'}$ 表示编码特征，则

$$
E_{nt}= \text{ClinicalBERT}(I_{nt})
$$

### 融合

受多模态自适应门（MAG）的启发，我们通过注意力门融合三种模式，如下图：

![](mag.png)

> MAG 在视觉/音频和文本模态之间计算一个位移向量H通过跨模态自注意力。此操作在单词级别上执行，以便根据非语言线索调整单词表示。
>
> 然而，与他们工作中的视频数据不同，我们任务中的多模态数据本质上是非同步的，这意味着每个词都没有伴随的模态。此外，与上述情感分析任务相比，text/notes 在医学预测中或许更重要。

我们在样本级别做多模态融合，并根据需要切换主要模式。图中 Attention Gating 的计算如下，其中，$g \in \mathcal{R}$ 是门控值：

$$
g_1=\text{ReLU}(\text{Linear}([E_{nt};E_{ti}]))
$$

$$
g_2=\text{ReLU}(\text{Linear}([E_{nt};E_{ts}]))
$$

位移矢量H是由Eti和Ets乘以各自的门控值后相加得到的，$H∈R^{B×D_3'}$：

$$
\bold{H}=\text{Linear}([g_1E_{ti};g_2E_{ts}])
$$

根据临床记录的主要模态，对主要特征Ent和位移向量H进行加权求和以创建多模式表示M：
$$

\bold{M}=E_{nt}+\alpha\bold{H}
$$

$$
\alpha=\min(\frac{||E_{nt}||_2}{||\bold{H}||_2}\beta，1)
$$

其中β 是随机初始化的超参数，用于训练模型。使用了 Ent 和 H 的 L2 范数。缩放因子 α 用于限制偏移量向量 H 在期望范围内的影响。

### 预测

用softmax层为多标签问题（如诊断预测）生成预测，其中，$\hat{Y}∈\mathcal{R}^{B×N}$ ， N 是标签的数量：

$$
\bold{\hat{Y}}=\text{softmax}(\text{Liner}(\bold{M}))
$$

对于二分类问题，我们使用 Sigmoid 层代替 Softmax，其中，$\hat{y}∈\mathcal{R}^{B}$ ：

$$
\bold{\hat{y}}=\text{sigmoid}(\text{Liner}(\bold{M}))
$$

我们使用真实标签与预测之间的交叉熵来计算上述两个问题中所有入住ICU的损失。给定由Y提供的多标签问题的真实标签，计算如下：

$$
\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\bold{Y}_i\text{log}(\bold{\hat{Y}})+(1-\bold{Y}_i)\text{log}(1-\bold{\hat{Y}})
$$

## 实验

### 数据集

我们使用MIMIC-III数据集。我们关注的是 2008 年至 2012 年间使用 iMDSoft MetaVision 系统记录的 17,710 名患者（23,620 次 ICU 住院）的 17,710 名患者。

在我们的实验中，使用FIDDLE提取48/12小时内的非文本特征。它们以 7：1.5：1.5 的比例随机拆分为训练、验证和测试集。

48/12 小时内的文本特征由将每个类别的最新注释合并到一个文档中并使用 WordPiece标记。

由于时间限制和模态要求，我们排除了入住 ICU 少于 48/12 小时的患者以及那些有错误或没有注释的患者。经过数据预处理后，样本数量为 10,210/14,174。

### 诊断任务

使用自入住ICU起48小时内产生的数据来预测诊断。这是一个多标签问题，因为每次就诊可能与多种疾病有关。该标签是由将相应的国际疾病分类第九版（ICD-9）诊断代码转换为多热向量生成的。

在MIMIC-III的ICD-9定义表中提取ICD-9的前三位数字，得到1,042种疾病组。我们使用Top-k召回来评估此任务，因为它模仿了医生进行鉴别诊断的行为，即列出最有可能的诊断并相应地治疗患者以识别患者状况。在我们的实验中，我们将k分别设置为10、20和30。

### ARF

使用患者入住ICU后12小时内产生的数据来预测患者是否会患上急性呼吸衰竭（ARF）。这是一个二分类问题。我们使用AUROC(Area Under the Receiver Operating Characteristic curve)和AUPR(Area Under the Precision-Recall curve)来评估。

注意到我们的诊断任务是一项全新的任务，我们认为，在临床实践中，越早做出诊断，其价值就越高。因此，我们在这项工作中提取了前 48/12 小时的数据来进行诊断预测，而不是整个入院期间的数据。

于是我们的模型输入不同于以前的一些工作，例如，医疗代码是他们的输入数据的重要组成部分，而它们尚未在时间限制内生成，因此未包含在我们的输入数据中。鉴于这些模型是专门为它们的输入设计的，我们无法将其性能与我们的模型进行比较。因此，基准线将其排除在外。

### Baseline

![](baseline_MI.png)

上图是一些基线模型表现，命名规则如下：

- 前缀 F 表示时间无关和时间序列数据的融合。
- LR 表示逻辑回归，RF 表示随机森林。
- 后面的每个模型名称可以分为两个模块，第一个模块表示融合中的主要模式。
- 第二部分是Clinical-BERT与LSTM和CNN进行的融合。
- 第三部分是Clinical-BERT与Star-Transformer和Transformer编码器进行的融合。

### 实验设计

具体来说，我们在本论文中训练的所有模型都使用Adam进行训练，学习率为1e−4。每个模型的dropout设置为0.1。对于Clinical-BERT，我们采用BERT基础架构的默认配置并加载预训练的参数以进行微调。

- 所有模型编码时间无关数据的维度均设置为64，即$D_1'=64$。
- 在ARF任务中，性能最好的模型是LstmBert，在此我们设置 $L_2'=512, D_2'=128$

我们使用单个Lstm层。LstmBert中有120M个参数。在诊断任务中，最佳模型是BertEncoder，其中编码器的隐藏大小和层数分别设置为1024和3。BertEncoder中有150M个参数。

### 结果

- 在ARF任务中，我们观察到所有四个时间序列编码器都显著优于BERT。在这个任务中，时间序列数据比临床记录更有效。
- 在诊断任务中，BERT 在四种时间序列编码器上显著地表现得更好。相反的趋势表明，在这个任务中临床笔记应该是主要模式。