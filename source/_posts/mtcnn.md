---
title: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
date: 2018-04-25 12:56:33
categories: CNN
tags: 
  - object detection
  - paper
  - face detection
image: /images/mtcnn/pipeline.png
---

> 提出一种深度级联的多任务框架，利用检测和对齐的固有相关性去增强它们的性能。实际中，利用有三阶段精细设计的深度卷机网络的级联结构，由粗到精地检测和对齐人脸。

<!-- more -->

# Introduction
人脸识别中视觉的变化，比如遮挡，姿态变化和极端的光照条件，会给人脸检测和对齐带来巨大挑战。
`AdaBoost`和`Haar-Like`特征训练的级联分类器虽然可以达到比较高的效率，但是大量研究表明这类检测器在人脸有着较大的视觉变化时，检测精度会大大降低。`DPM（deformable part models）`用于人脸检测也可以达到非常好的性能，然而计算代价太大，并且在训练时可能要求大量的标注。

人脸对齐领域的方法可以大致划分为两类：基于回归的方法和模板匹配方法。过去大部分的人脸检测和对齐方法都忽视了这两种任务之间的固有联系。

另一方面，挖掘难样本对于增强检测器的性能是至关重要的。传统的方法都是离线模式去挖掘，对于人脸检测任务来说，需要一种在线的难阳本挖掘方法，这样可以自动适应当前的训练状态。

本文中，通过多任务学习使用统一的级联`CNNs`集成这两种任务。提出的`CNNs`包含三个阶段：第一阶段，使用浅层的`CNN(fast Proposal Network (P-Net))`生成候选窗口；第二阶段，通过更加复杂的`CNN(Refinement Network (R-Net))`去精炼窗口，拒绝掉大量的非人脸的窗口；第三阶段，使用更加强大的`CNN(Output Network (O-Net))`去再次精修结果并输出`5`个`landmark`位置。

<center><img src="/images/mtcnn/pipeline.png" width = "50%"/></center>

贡献：
- 提出一种级联的`CNNs`框架做人脸检测和对齐，设计了一种轻量的`CNNs`结构用于实时性能。
- 提出一种在线难样本挖掘（`online hard sample mining`）方法去提高性能。

# Approach

## Overall Framework
首先将给定图像缩放到不同的尺度建立图像金字塔，这将是后面三阶段级联框架的输入。

- 第一阶段：采用全卷积神经网络，即`P-Net`，去获得候选窗体和`bounding box`回归向量。同时，候选窗体根据估计的`bounding box`向量进行校准。然后，利用`NMS`方法合并高度重叠的候选框。

- 第二阶段：所有的候选框被输入`R-Net`，进一步拒绝掉大量的错误候选框，同样使用`bounding box`回归校正候选框，并实施`NMS`。

- 第三阶段：和第二阶段相似，但是目的是利用更多的监督去判断人脸区域，并输出`5`个`landmark`位置。

## CNN Architectures
多个`CNN`被用于人脸检测，但其性能可能受到以下情况的限制：
- 卷积层中的卷积核缺乏多样性，限制他们的识别能力；
- 对比多类识别检测和分类任务，人脸检测是一个二分类问题，因此每一层需要的卷积核较少。所以本文减少卷积核数量，并将`5*5`的卷积核大小改为`3*3`的，以此在增加深度来提高性能的同时减少计算。
<center><img src="/images/mtcnn/Architectures.png" width = "80%"/></center>  

## Training
 本算法从三个方面对`CNN`检测器进行训练：人脸分类、`bounding box`回归、`landmark`定位（关键点定位）。

 **人脸分类：** 二分类问题，使用交叉熵损失函数：

 <center><img src="/images/mtcnn/Eq1.png" width = "50%"/></center>

 **`bounding box` 回归：** 回归问题，使用欧氏距离计算的损失函数：

<center><img src="/images/mtcnn/Eq2.png" width = "28%"/></center>

 **`landmark` 定位：** 回归问题，使用欧氏距离计算损失函数：

<center><img src="/images/mtcnn/Eq3.png" width = "50%"/></center>

在每个`CNN`中实现的是不同的任务，所以在学习过程中有几种类型的训练图像：人脸，非人脸，部分对齐的人脸。在这种场合下，上面的式子不能使用，比如，对于背景区域的样本，只需要计算检测损失，其他两种损失设置为`0`，所以使用一些系数，**总体的学习目标**表示为：

<center><img src="/images/mtcnn/Eq4.png" width = "50%"/></center>

**Online Hard sample mining：** 每一个`mini-batch`中，对从所有的样本前向运算得到的损失排序，选择前`70%`作为难样本。在反向传播中，只计算来自于这些难样本的梯度。这意味着在训练中忽视掉那些对增强检测器性能帮助甚小的简单样本。

# Experiments
在训练中有四种数据：
1. 负样本：与任何`ground truth faces`的`IoU`低于`0.3`的。
2. 正样本：与一个`ground truth face`的`IoU`高于`0.65`的。
3. `Part faces`：与一个`ground truth face`的`IoU`在`0.4~0.65`之间的。
4. `Landmark faces`：标定了`5`个`landmark`的。
负样本和正样本用于人脸分类，正样本和`part faces`用于`bounding box`回归，`landmark faces`用于`landmark`定位。上面的样本比例：`3：1：1：2`。

数据的收集方法如下：
- `P-Net`：随机地从`WIDER FACE`数据集中裁切一些图像块，收集正样本，负样本，`part`人脸。从`CelebA`数据库裁切人脸作为`landmark`人脸。

- `R-Net`：使用框架的第一阶段在`WIDER FACE`中检测人脸，收集正样本，负样本和`part`人脸，同时从`CelebA`中检测`landmark`人脸。

- `O-Net`：与`R-Net`相似的方法收集数据。但是是使用前两个阶段去检测人脸和收集数据。

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
