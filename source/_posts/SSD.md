---
title: SSD:Single Shot MultiBox Detector
date: 2018-04-15 10:12:05
categories: CNN
tags: 
  - object detection
  - paper
image: /images/SSD/Framework.png
---
> 这篇文章发表在`ECCV2016`上，在既保证速度，又要保证精度的情况下，提出了`SSD`。使用`single deep neural network`，便于训练与优化，同时提高检测速度。`SSD`将`bounding box`的输出空间离散化为一组默认框，这些默认框在`feature map`每个位置有不同的高宽比和尺度。在预测时，网络对每一个默认框中存在的目标生成类别分数，并且调整边界框以更好地匹配目标形状。除此之外，网络对不同分辨率的`feature map`进行组合，以处理各种尺寸的目标。
<!-- more -->

# Introduction
目前流行的`state-of-art`的检测系统大致都是如下步骤，先生成一些假设的`bounding boxes`，然后在这些`bounding boxes`中提取特征，之后再经过一个分类器，来判断里面是不是物体，是什么物体。这些方法计算时间长，即使有的速度提升了，却是以牺牲检测精度来换取时间的。

这篇文章提出了第一个在`bounding box`预测上不需要重新采样像素和特征的基于深度网络的目标检测器。这使得速度和精度都有了改善。**速度上的根本改进是因为消除了`bounding box proposal`和后续的像素或特征重采样阶段**。

改进包括：使用小的卷积核去预测目标类别和边界框位置的偏移，使用单独的预测器（`filter`）解决不同高宽比的检测，并且将这些`filter`应用到网络后期的`feature map`上去实现多尺度的检测。

**`SSD`的贡献**：
- 比`YOLO`更快，更准确；和`Faster R-CNN`一样准确。
- 核心是使用小卷积核来预测特征图上固定的一组默认边界框的类别分数和位置偏移。
- 为了实现高检测精度，从不同尺度的特征图产生不同尺度的预测，并且得到不同高宽比的预测。
- 这些设计实现了简单的端到端训练和高精度的检测，即使输入相对低分辨率图像，也能在速度和精度之间取得更好的权衡。

# The Single Shot Detector (SSD)
**以卷积形式，在不同尺度（例如`8×8`和`4×4`）的特征图中的每个位置上评估一组不同高宽比的默认框(`default box`)。 对于每个默认框，预测位置偏移和目标类别分数（`c1，c2，...，cp`）**。

在训练时，首先将这些默认框匹配到真实标签框。例如，两个蓝色虚线默认框匹配到猫，一个红色虚线框匹配到狗，这些框为正，其余为负。模型损失是位置损失（例如`Smooth L1`）和置信度损失之间的加权和。

- `feature map`尺寸例如是`8×8`或者`4×4`的
- `default box`就是每一个位置上，一系列固定大小的`box`，即图中虚线所形成的一系列 `boxes`。同一个`feature map`上`default box`的`aspect ratio`不同，不同的`feature map`上`default box`有着不同的`scale`。

<center><img src="/images/SSD/Framework.png" width="80%"/></center>

## Model
**`SSD`产生一组检测框和框中目标类别分数，接着使用非极大化抑制产生最终检测**。本文在基础网络后添加辅助结构，产生了具有以下主要特征的检测：

**Multi-scale feature maps for detection**：将卷积特征层添加到截断的基础网络的末尾。这些层尺寸逐渐减小，并且允许多个尺度的预测。检测的卷积模型对于每个特征层是不同的。

**Convolutional predictors for detection**：每个添加的特征层（或基础网络结构中的现有特征层）可以使用一组卷积核产生固定的预测集合。对于具有`p`个通道的大小为`m×n`的特征图，使用`3×3×p`卷积核卷积操作，产生类别分数或相对于默认框的坐标偏移。在应用卷积核运算的`m×n`个位置的每一处，产生一个输出值。`bounding box`偏移输出值是相对于默认框的，默认框位置则相对于特征图。

**Default boxes and aspect ratios**：对于网络顶层的多个`feature map`，将一组默认框与`feature map`每一个位置关联，每一个默认框的位置对于其在`feature map`中的位置是固定的。具体来说，**对于在给定位置的`k`个框中每个框，计算`c`个类别分数和相对于原始默认框的`4`个偏移量。所以在特征图中的每个位置需要总共`（c+4）k`个卷积核，对于`m×n`特征图产生`（c+4）kmn`个输出**。这里的默认框类似于`Faster R-CNN`中使用的`anchor boxes`，`Faster R-CNN`将`anchor`只用在基础网络的最后一个卷积层的输出上，但本文将其应用在不同分辨率的`feature map`中。在多个特征图中使用不同的默认框形状，可以有效地离散可能的输出框形状空间。

**SSD与YOLO网络结构图对比如下**：

`SSD`模型在基础网络的末尾添加了几个特征层，这些层预测了对于不同尺度和高宽比的默认框的偏移及其相关置信度。`YOLO`是用全连接层预测结果，而`SSD`是用卷积层预测结果。
<center><img src="/images/SSD/SSDandYOLO.png" width="90%"/></center>
 
 从上图中可以看出，`SSD`使用`VGG-16`作为`basenet`，`conv4_3`输出的`feature map`也被进行卷积操作预测结果，在`VGG16`的`conv5_3`后，用卷积层替代了全连接层，然后是`SSD`增加的额外的`feature layer`。**最后网络输出了`8732`个候选框信息**（`38x38x4+19x19x6+10x10x6+5x5x6+3x3x4+1x4`）。不同层级，不同尺度的`feature map`用作预测，所有的这些预测又经过了非极大值抑制得到最终结果。


## Training
 在训练时，`SSD`与那些用`region proposals + pooling`方法的区别是：`SSD`训练图像中的`ground truth`需要被匹配到`default boxes`上。如上面的图中，狗的`ground truth`是红色的`bounding boxes`，在进行标注的时候，要将红色的 `ground truth box`匹配到图（`c`）中一系列固定输出的`boxes`中的一个，即图（`c`）中的红色虚线框。`SSD`的输出是事先定义好的，一系列固定大小的`bounding boxes`。
 


**Matching strategy**：对于每一个`ground truth box`，选择一些位置不同，高宽比以及尺度不同的`default box`。首先把每个`ground truth box`与和它具有最高`IoU`的默认框匹配，确保每个真实标签框有一个匹配的`default box`。然后把这些`default box`与和它`IoU`高于阈值（`0.5`）的任何`ground truth box`匹配起来。添加这些匹配简化了学习问题：当存在多个重叠的`default box`时，网络可以预测多个较高的分数，而不是只能选取一个具有最大`IoU`的框。

**Choosing scales and aspect ratios for default boxes**： 大部分`CNN`网络在越深的层，`feature map`的尺寸会越来越小。这样做不仅仅是为了减少计算与内存的需求，还有个好处就是，最后提取的`feature map`就会有某种程度上的平移与尺度不变性。为了处理不同尺度的物体，这些网络将图像转换成不同的尺度，然后独立的通过`CNN`网络处理，再将这些不同尺度的图像结果进行综合。**本文通过用单个网络中的若干不同层的特征图来进行预测，可以处理不同尺度的目标检测，同时还在所有目标尺度上共享参数**。一些文献表明使用来自较低层的特征图可以提高语义分割质量，因为较低层保留的图像细节越多。添加从高层特征图下采样的全局上下文池化可以帮助平滑分割结果。受这些方法的启发，使用低层和高层的特征图进行检测预测。

网络中不同层级的特征图具有不同的感受野大小。这里的感受野，指的是输出的`feature map`上的一个节点，其对应输入图像上尺寸的大小。在`SSD`框架内，默认框不需要对应于每层的实际感受野。我们可以设计默认框的平铺，使得**特定特征图，学习响应特定尺度的目标**。

 通过组合多个特征图在每个位置不同尺寸和宽高比的所有默认框的预测，得到了具有多样化的预测集合，覆盖各种目标尺寸和形状。例如狗被匹配到`4×4`特征图中的默认框，但不匹配到`8×8`特征图中的任何默认框。这是因为那些默认框具有不同的尺度但不匹配狗的`bounding box`，因此在训练期间被认为是负样本。

**Hard negative mining**：在匹配步骤之后，大多数默认框都是负样本，特别是当可能的默认框数量很大时。这导致了训练期间正负样本的严重不平衡。所以按照每个默认框的最高置信度对它们进行排序，并选择前面的那些，使得负样本和正样本之间的比率最多为`3：1`，以代替使用所有的负样本。这导致更快的优化和更稳定的训练。

# Result
<center><img src="/images/SSD/result.png" width="70%"/></center>

# Reference
1. [论文原文](https://arxiv.org/abs/1512.02325)
2. [论文阅读：SSD](http://blog.csdn.net/u010167269/article/details/52563573)
3. [SSD算法及Caffe代码详解](http://blog.csdn.net/u014380165/article/details/72824889)

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
