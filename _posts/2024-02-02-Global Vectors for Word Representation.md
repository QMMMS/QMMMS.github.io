---
title: GloVe：Global Vectors for Word Representation
date: 2024-02-02 8:21:00 +0800

img_path: "/assets/img/posts/2024-02-02-Global Vectors for Word Representation"
categories: [深度学习]
tags: [读论文]
math: true
---

> 读论文时间！
>
> 词嵌入模型：GloVe
>
> 参考：
>
> - [https://blog.csdn.net/qq_44579321/article/details/128120877](https://blog.csdn.net/qq_44579321/article/details/128120877)
> - [https://blog.csdn.net/qq_22795223/article/details/105737651](https://blog.csdn.net/qq_22795223/article/details/105737651)
{: .prompt-info }

## 介绍

glove是斯坦福大学的一个开源项目，于 2014 年推出，它是一个基于全局词频统计（count-based & overall statistics）的词表征（word representation）工具**。**该模型是包含词的向量表示的无监督学习算法。这是通过将单词映射到有意义的空间来实现的，其中单词之间的距离与语义相似性有关。

Glove结合了LSA和word2vec两者的优点。

- LSA是一种词向量表征工具，也是基于共现矩阵，采用了基于奇异值分解SVD的矩阵分解技术对大矩阵进行降维，但是SVD计算代价太高，并且它对于所有单词的统计权重都是一致的，而glove克服了这些缺点。
- Word2vec因为它是基于局部滑动窗口计算的，利用了局部上下文的特征，所以缺点就是没有充分利用所有的语料。  

![](window.gif)

> Word2vec是一种有效创建词嵌入的方法，用一句比较简单的话来总结，word2vec是用一个一层的神经网络把one-hot形式的稀疏词向量映射称为一个n维(n一般为几百)的稠密向量的过程。
>
> word2vec里面有两个重要的模型：
>
> - CBOW模型(Continuous Bag-of-Words Model)，根据某个词前面的C个词或者前后C个连续的词，来计算某个词出现的概率。
> - Skip-gram模型，是根据某个词，然后分别计算它前后出现某几个词的各个概率。

Glove的创新点在于它是一种新的词向量训练模型，并且能够在多个任务上取得最好的结果。相对于原始概率，概率的比值更能够区分相关和不相关的词。

思想为：首先建立单词-单词共现矩阵，矩阵中的每一个元素代表每个单词在相应上下文环境中共同出现的次数，利用共现次数去计算共现比例，建立词向量和共现矩阵之间的映射关系，训练得到glove模型。再去建立损失函数，用一些优化算法最终得出词向量。

## 模型

### 共现矩阵

根据语料库（corpus）构建一个共现矩阵（Co-occurrence Matrix）X，矩阵中的每一个元素  $X_{i j}$ 代表单词  i  与上下文单词 j 在特定大小的上下文窗口（context window）内共同出现的次数。

一般而言，这个次数的最小单位是 1 ，但是在文章中指出，根据两个单词在上下文窗口的距离 d ，提出了一个 衰减函数 $decay=\frac{1}{d}$ 用于计算权重，也就是说距离越远的两个单词所占总计数的权重越小。

**例如**：语料库如下：

- I like deep learning.
- I like NLP.
- I enjoy flying.

则窗口为1的共现矩阵如下：

![](gxjz.png)

按照作者的描述，实验中计算的不是真正意义上的共现次数，而是共现次数和权重递减函数的乘积，从而达到距离越远的共现词对权重越低，距离越近的共现词对权重相对较大。

### 利用共现矩阵求解概率

$$
P_{ik}=P(k|i)=\frac{X_{ik}}{X_i}
$$

$$
X_i = \sum_{j=1}^VX_{i,j}
$$

其中：

- X 为共现矩阵
- i 为中心词语
- k 为上下文词语
- $X_{ik}$ 为中心词语与上下文词语共现次数
- $P_{ik}$ 为中心词语周围出现上下文词语的概率
- V是词汇表的大小（共现矩阵维度为V* V）

然后我们从一个语料库中拿那些词来举个例子：

![](glove_3.png)

可以发现，比值P(k∣ice)/P(k∣steam)在一定程度上可以反映词汇之间的相关性，当k 与 ice 和 steam相关性比较低时，其值应该在1附近，当 k 与 ice 或者 steam 其中一个相关性比较高时，比值应该偏离1比较远。

基于这样的思想，作者提出了这样一种猜想，能不能通过训练词向量，使得词向量经过某种函数计算之后可以得到上面的比值，具体如下：

$$
F\left(w_{i}, w_{j}, \tilde{w}_{k}\right)=\frac{P_{i k}}{P_{j k}}
$$

其中，w 是词向量，F是未知函数，通过将 F 限定为指数函数，再加入偏差项标量 b，经过变换，构建词向量（word vector）和共现矩阵（Co-occurrence Matrix）之间的近似关系：

$$
w_i^T\widetilde{w}_j+b_i+\widetilde{b}_j=log(X_{ij})
$$

### 损失函数

此时模型的目标就转化为通过学习词向量的表示，使得上式两边尽量接近，因此，可以通过计算两者之间的平方差来作为目标函数，即：

$$
J = \sum_{i, j=1}^{V}(w_{i}^{T} \tilde{w}_{j}+b_{i}+ \widetilde{b}_{j}-\log X_{i j})^{2}
$$

但是这样的目标函数有一个缺点，就是对所有的共现词汇都是采用同样的权重，因此，作者对目标函数进行了进一步的修正，通过语料中的词汇共现统计信息来改变他们在目标函数中的权重，具体如下：

$$
J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2}
$$

其中，f是权重函数。权重应该满足下面这些条件：

- 这些单词的权重大于那些很少出现在一起的单词（rare co-occurrences），所以这个函数是非减函数（non-decreasing）
- 但是我们也不希望这个权重过大（overweighted），当到达一定程度后应该不再增加，这样才不会出现过度加权。
- 如果两个单词没有在一起出现，也就是 $X_{ij} = 0$ ,那么它们不应该参与到损失函数的计算中去，也就是要 f(x) 满足 f(x)=0

据此，作者选择了这个函数：

$$
f(x)=
\left\{\begin{array}{ll}
(x / x_{\max })^{\alpha} & \text { if } x<x_{\max } \\
1 & \text { otherwise } 
\end{array}\right.
$$

作者在实验中设定 $x_{\max }=100$ ，并且发现 $\alpha=3 / 4$时效果比较好，如下图：

![](glove_f.png)

## 实验

虽然很多人声称GloVe是一种无监督(unsupervised learning)的学习方式（因为它确实不需要人工标注label），但其实它还是有label的，这个label就是公式中的 $log(X_{ij})$，而公式中的向量w和$\tilde{w}$就是要不断更新/学习的参数。

$$
J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2}
$$

所以本质上它的训练方式跟监督学习的训练方法没什么不一样，都是基于梯度下降的。具体地，这篇论文里的实验是这么做的：采用了AdaGrad的梯度下降算法，对矩阵X中的所有非零元素进行随机采样，学习曲率(learning rate)设为0.05，在vector size小于300的情况下迭代了50次，其他大小的vectors上迭代了100次，直至收敛。

最终学习得到的是两个vector ，w和$\tilde{w}$，因为X是对称的(symmetric)，,所以从原理上讲w和$\tilde{w}$是也是对称的，他们唯一的区别是初始化的值不一样，而导致最终的值不一样。所以这两者其实是等价的，都可以当成最终的结果来使用。

但是为了提高鲁棒性，我们最终会选择两者之和 $w+\tilde{w}$ 作为最终的vector，因为两者的初始化不同相当于加了不同的随机噪声，所以能提高鲁棒性。