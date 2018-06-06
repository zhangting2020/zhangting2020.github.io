---
title: Feature Pyramid Networks for Object Detection
date: 2018-05-06 21:00:21
categories: CNN
tags:
  - object detection
  - instance segmentation
  - feature pyramid
  - paper
image: /images/FPN/pyramid2.png
---

> 这篇文章提出特征金字塔网络（FPN），将分辨率高语义性弱的浅层特征和分辨率低语义性强的深层特征融合，形成了多级金字塔，在金字塔每一级上独立检测目标。FPN不仅对多尺度的目标检测具有很好的效果，还可以应用到分割任务中。

<!-- more -->

# Introduction
识别不同尺度的目标是计算机视觉的一项基本挑战，下面介绍四种利用特征的形式：
- **图像金字塔**：构建在图像金字塔上的特征金字塔（简称为特征化的图像金字塔），如图（a）。这种情况下，图像被采样为多种尺度，然后生成不同尺度的特征。DMP就是使用密集的尺度采样获得了不错的效果。这种方法被大量用在手工设计的特征中。优点是：每一级的特征语义信息都比较强，缺点是预测时间长。在图像金子塔上端到端地训练深度卷积神经网络是不切实际的，因为内存消耗大，所以即使要用，也只用在做预测时。
- **利用卷积神经网络提取特征，使用最后一层的特征做预测**。卷积神经网络可以提取高级的语义表达，对尺度变化有更好的鲁棒性，所以可以使用单尺度特征，如图（b）。
- **使用不同层的特征进行预测**，如图（c）。SSD使用卷积神经网络多个层的特征分别做预测，如同一个特征化的图像金字塔。SSD重复利用前向传播过程中计算好的不同层特征，所以几乎没有带来额外代价。但SSD没有利用到足够底层的特征，因为底层特征语义信息弱，但是底层特征分辨率高，对检测小目标很重要。
- 本文提出的FPN，**将低分辨率语义信息强的浅层特征与高分辨率语义信息弱的深层特征进行组合，构建特征金字塔，在金字塔的每一级上分别做预测**，如图（d）。
<center><img src = "/images/FPN/pyramid.png" width = 80%></center>

还有一种相似的结构，采用自顶向下的方法和跳跃连接（skip connection），目的是生成单个具有较好分辨率的高级特征图，然后在这个特征图上做预测，如下面上半部分的图。本文与其结构很接近，但是利用它形成一个金字塔，在金字塔的每一级独立地进行预测。
<center><img src = "/images/FPN/pyramid2.png" width = 40%></center>

# Feature Pyramid Networks
FPN接受一个具有任意尺寸的单尺度图像作为输入，在多个层级以全卷积的形式生成不同尺寸比例的特征图。**FPN的构建涉及三个部分：自底向上的路径（Bottom-up pathway），自顶向下的路径（top-down pathway），横向连接（lateral connection）**。

## 自底向上的路径
自底向上的路径就是主干网络（backbone）的前向计算，产生不同尺度的feature map，尺度的比例为2，即每次下采样都是缩小2倍。**自底向上的过程，空间分辨率降低，但是语义性增加**。

这里把那些会输出同样尺寸feature map的层归为一个stage。每个stage都定义了一级金字塔，每个stage的最后一层特征被选取出来。本文的backbone是ResNet，选取的是每一个stage最后一个残差块的输出，将Conv2，Conv3，Conv4，Conv5的输出定义为{C2，C3，C4，C5}。这些输出相对于输入图像，stride为{4，8，16，32}，即分辨率缩小的倍数。为啥不用C1呢，因为维度太高了，内存消耗大。

## 自顶向下和横向连接
FPN通过自顶向下的方式，**从语义信息丰富的层出发，构建出分辨率更高的层，将这些层的特征与浅层的特征通过横向连接融合**。如下图，对于空间分辨率较粗糙的深层特征，进行2倍的上采样（最近邻），相应的浅层特征使用1x1的卷积降维，之后与上采样的特征通过逐元素相加合并。合并后的特征又通过3x3的卷积生成最终的feature map，即{P2，P3，P4，P5}，这一步降低了上采样的混叠效应（aliasing effect）。
<center><img src = "/images/FPN/top-bottom.png" width = 45%></center>

FPN可以更加详细的用下图表示：
<center><img src = "/images/FPN/feature.png" width = 60%></center>

金字塔中所有的层都共享分类器和回归器，就像传统的图像金字塔那样。固定特征的通道数为256，因此所有额外的层输出都为256通道。这些额外的层没有使用非线性。

# Applications
## FPN for RPN
回顾一下Faster R-CNN中的RPN：
- 预先定义了一组不同尺度和高宽比的anchor，覆盖不同形状的目标。
- 在最后一个单尺度的共享卷积层输出的feature map上，使用一个3×3的滑动窗（卷积核），随后是两路1x1的卷积，分别用来分类是否为目标以及回归bounding box，这里将这一部分称为网络的头（head）。

将FPN应用在RPN上的要点：
- 将单尺度的feature map替换为FPN，为每一级金字塔附加一个head，也是3x3的卷积和两路1x1的卷积。
- 每一级都是单尺度的anchor。因为网络头需要在每层金字塔的feature map上的所有位置滑动，就没必要在特定的一级使用多尺度的anchor了。{P2，P3，P4，P5}上的anchor面积分别为{32x32，64x64，128x128，256x256，512x512}。
- 跟随Faster R-CNN，每一级金字塔使用多个高宽比{1:2，1:1，2:1}。所以金字塔上一共是15种anchor：5种尺度x3种高宽比.
- 与Faster R-CNN一样，正样本是与ground truth有着最高IoU的，以及与任何ground truth有着高于0.7的IoU的anchor，与所有ground truth的IoU都小于0.3的anchor作为负样本。注意：ground truth是与anchor相关的，因此也就与金字塔的某一级相关了。

RPN头在金字塔的所有层上共享参数，作者做了不共享参数的实验，发现性能相似，说明FPN中金字塔的所有层共享相似的语义级别。

## FPN for Fast R-CNN
Faster R-CNN另一模块是Fast R-CNN中基于region的检测器，它使用RoI Pooling提取特征，也是在一个单尺度feature map上进行预测。
<center><img src = "/images/FPN/Faster-R-CNN.jpeg" width = 100%></center>

应用FPN时，使用前面一节描述的RPN生成多个感兴趣区域RoI，根据RoI在原始图像中的尺寸，选择尺度最正确的feature map，去提取这个RoI的feature。
<center><img src = "/images/FPN/Faster-R-CNN.jpeg" width = 100%></center>

若RoI在原始图像中高宽为w和h，那么对应的特征层为
<center><img src = "/images/FPN/eq1.png" width = 30%></center>

224是标准的ImageNet预训练尺寸，$k_0$是当RoI的尺寸为224x224时，对应的特征级。使用ResNet为基础网络的Faster R-CNN，使用C4作为后续部分的输入feature，所以$k_0=4$。直觉上，公式表示当RoI的尺度变小时，比如变为112，那么它应该被映射到一个分辨率更精细的层（$k=3$）。

# Extensions: Segmentation Proposals
FPN还可以扩展到分割任务中，下面使用FPN生成分割的proposal。特征金字塔的构建方式与应用FPN到目标检测中一样，但是维度设为128，原来是256。以全卷积的形式，使用5x5的滑动窗在特征图上滑动，产生14x14的掩码和目标分数。浅橙色是相应的图像尺寸，深橙色是目标尺寸，可以看出掩码也是在金字塔的不同级独立预测的。
<center><img src = "/images/FPN/seg.png" width = 70%></center>

# Reference
1.[Understanding Feature Pyramid Networks for object detection (FPN)](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)

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


