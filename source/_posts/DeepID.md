---
title: DeepID:Deep Learning Face Representation from Predicting 10,000 Classes
date: 2018-04-10 11:04:01
categories: CNN
tags:
  - face recognition
  - paper
image: /images/DeepID/FacePatch.png
---

> 这篇文章提出使用深度学习去学习到一个高级的特征表达集合`DeepID`用于人脸验证。`DeepID`特征是从深度卷积神经网络的最后一个隐含层神经元激励提取到的。并且这些特征是从人脸的不同区域中提取的，用来形成一个互补的过完备的人脸特征表达。
<!-- more -->

# Introduction
当前有着最优表现性能的人脸验证算法采用的是过完备的低级别特征，并且使用的是浅层模型。本文提出使用深度模型来学习高级的人脸特征集，也就是，把一个训练样本分入`10000`个身份中的一个。高维空间的操作虽然更有难度，但学习到的特征表达有更好的泛化性能。尽管是通过识别任务学习的，但是这些特征也可用于人脸验证或者数据集中没有出现过的人脸。

特征提取过程如下：卷积神经网络通过学习，将训练集中所有人脸根据他们的身份进行分类。使用多个`ConvNet`，每一个提取最后一个隐含层神经元的激励作为特征（`Deep hidden IDentity features, DeepID`）。每一个`ConvNet`取一个人脸`patch`作为输入并且提取底层（`bottom layers`）的局部低级特征（`low-level`），随着更多全局的高级特征逐渐在顶层（`top layer`）形成，特征的数量沿着特征提取级联（`feature extraction cascade`）持续减少。在级联的最后形成了一个高度紧凑的`160-d DeepID`特征，它包含了丰富的身份信息，用来预测一个数量庞大的身份类别。
<center><img src="/images/DeepID/Feature_Extract.png" width="70%"/></center>

分类所有的身份而不是训练二分类基于两点考虑
- 分类训练样本到多类别中的一个比起实现二分类更加困难。这样可以充分利用神经网络强大的学习能力提取有效的人脸特征用于识别。
- 多分类给卷积神经网络潜在地增加了强正则化，这有助于形成共享的隐含的特征表达，更好地分类所有的身份。

限制`DeepID`的维度明显小于要预测的类别数，是学习高度紧凑的具有辨识力特征的关键。进而拼接从不同的人脸区域提取到的`DeepID`形成互补的过完备特征表达。

在仅使用弱对齐的人脸的情况下，在`LFW`数据集上达到了`97.45%`的人脸验证准确率。同时也观察到，随着训练身份数量的增加，验证的性能也在稳步提升。

# Deep ConvNets
这里的卷积神经网络共有`4`个卷积层，前`3`个每个都跟随着最大池化。卷积层之后是全连接的`DeepID layer`和`softmax layer`。`DeepID layer`是固定的`160`维，输出层`softmax`的维度随着预测类别数目而变化。

最后一个隐含层通过全连接与第三个和第四个卷积层相连，目的是可以得到多尺度的特征，因为第四个卷积层的特征比第三个更加的全局。这对于特征学习很关键，随着级联中连续的下采样，第四个卷积层包含了太少的神经元，变成了信息传播的瓶颈（`bottleneck`），所以在第三个和最后一个隐含层之间加入这种旁路连接（`bypassing connection`），即跳过某些层，最后一个隐含层减少了第四个卷积层可能的信息损失。
<center><img src="/images/DeepID/ConvNet.png" width="70%"/></center>

# Feature extraction
人脸对齐部分只用了`5`个点，两个眼睛的中心，鼻尖，两个嘴角。从人脸图像中选取`10`个区域，`3`种尺度，`RGB`和灰度图像一共`60`（`10x3x2`）个人脸`patch`来做特征提取。下图展示了`10`个人脸区域和具有`3`种尺度的两个特定人脸区域。
<center><img src="/images/DeepID/FacePatch.png" width="70%"/></center>

训练`60`个`ConvNet`，每一个`ConvNet`都从特定的`patch`和它的水平翻转中提取`160`维的`DeepID`向量。除了两眼中心和两个嘴角附近的`patch`不用自身翻转，而是使用对称的`patch`，也就是说左眼中心的`patch`的翻转是通过翻转右眼中心的`patch`得到的。这样每个`ConvNet`得到的整个`DeepID`的长度是$160\times2\times60$。


## Face verification
使用联合贝叶斯（`Joint Bayesian`）方法进行人脸验证，同时训练了一个神经网络做对比试验。输入层接受`DeepID`特征，输入的特征被分为`60`组，每一组包含了使用特定的`ConvNet`和`patch`对的`640`个特征（一个`patch`是`320`维的，验证需要两个人脸图像，因此就是一对`patch`）。同一组的特征是高度相关的。

局部连接层（`locally-connected layer`）的神经元只和一组特征连接，学习它们的局部相关性，同时降维。第二个隐含层式全连接层，学习全局的联系。输出只有`1`个神经元，以全连接的方式和前一层相连。隐含层用的激活函数是`ReLU`，输出层时`sigmoid`。

由于对输入的神经元不能使用`dropout`，因为输入的是高度紧凑的特征，并且是分布式表达，必须共同使用从而表达身份。但是高维的特征如果不用`dropout`会容易出现梯度弥散的问题。所以先训练如下图所示的`60`个子网络，每一个都以单组特征作为输入。然后使用子网络第一层的权重初始化原始网络中对应的部分，再微调原始网络第二和第三层。
<center><img src="/images/DeepID/FaceVerification.png" width="70%"/></center>

## Experiments
数据集：在`CelebFaces`数据集（`87628`幅图像，每个人平均`16`张）上训练模型，在`LFW`上进行测试。训练时，`80%`的数据即`4349`张用来学习`DeepID`，使用剩下的学习人脸验证模型。

人脸验证：在学习联合贝叶斯模型之前，使用`PCA`降维将特征维度降为`150`维。验证的性能在很大的维度范围内都可以保持稳定。

- `Multiscale ConvNets`
如前面所示，将第三个卷积层最大池化后的部分直接连接到最后一个隐含层，通过去掉这个连接和带上这个连接进行比较，发现准确率从`95.35%`提升到了`96.05`。
- `Learning effective features`
指数级增加身份类别，通过`top-1`误差率观察分类能力，根据测试集的确认精确度观察学习到的隐藏层特征的性能。发现同时对大量的身份进行分类对学习到具有判别性和紧致的隐藏特征是关键。
- `Over-complete representation`
为了评估少个`patch`的组合对性能的贡献大。分别选择了`1,5,15,30,60`个`patch`组合的特征训练人脸验证模型。结果表明：提取更多的特征，性能更好。

# Reference
1. [论文原文](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)

<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: '<%= page.date %>', // 可选。默认为 location.href  比如我本人的就删除了
  owner: 'zhangting2020',              //比如我的叫anTtutu
  repo: 'GitComment',                 //比如我的叫anTtutu.github.io
  oauth: {
    client_id: '60737b1014bda221b290',          //比如我的828***********
    client_secret: 'ce34df0ac4253419bfaa84df9363844ed0e6f9b8',  //比如我的49e************************
  },
})
gitment.render('container')
</script>
