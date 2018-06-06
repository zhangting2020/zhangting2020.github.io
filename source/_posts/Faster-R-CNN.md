---
title: Faster R-CNN:Towards Real-Time Object Detection with Region Proposal Networks
date: 2018-03-01 17:52:52
categories: CNN
tags: 
  - object detection
  - paper
image: /images/Faster-R-CNN/architecture.png
---
> 这篇论文发表在`NIPS2015`上。`region proposal`计算出现瓶颈，因此引入`RPN`网络，与检测网络共享整幅图像的的卷积特征，所以`region proposal`几乎是没有代价的。`Faster R-CNN`能够达到实时，并且精确度高。
<!-- more -->

# Introduction
`R-CNN`实际扮演的是分类器，它并不能预测出检测框，只能对检测框进行精修，因此它的准确度主要取决于`selective search`部分。在之前的方法中`proposal`是预测时间的瓶颈。

本文使用深度卷积神经网络`RPN(Region Proposal Network)`计算`proposal`。在卷积特征的顶部，通过增加一些额外的卷积层，能够在一个规则网格上的每个位置同时回归`bounding box`并且给出目标得分。`RPN`是一种全卷积网络，并且可以端到端地训练从而生成`proposal`。


网络结构如下：
<center><img src="/images/Faster-R-CNN/architecture.png" width="40%"/></center>

与`R-CNN`，`SPP-Net`，`Fast R-CNN`相比，`Faster R-CNN`的主要不同在于，不是在原始图像上提取`proposal`，而是**先对原始图像进行卷积得到`feature map`，然后利用`RPN`在`feature map`上提取`proposal`**。
# Faster R-CNN
`Faster R-CNN`由两个模块构成：用于生成`region`的全卷积网络`RPN`；`Fast R-CNN`检测器。`RPN`模块告诉`Fast R-CNN`检测器应该“看”哪里。


## Region Proposal Networks (RPN)
**`RPN`采用任意尺寸的图像作为输入，然后输出一系列的带有目标得分（是否有目标）的`proposal`。这一过程使用一个全卷积网络**。
- 为生成`region proposal`，在最后一个共享的卷积层输出的`feature map`上使用一个`n×n`的滑动窗（本文`n=3`）
- **在每一个滑动窗的位置，同时预测多个`region proposal`**，每个位置最大可能的`region proposal`个数定义为`k`，文中是`9`。`reg`层输出就为`4k`，`cls`层输出为`2k`个分数，对应于每一个`region proposal`为目标或非目标的概率。
- 每一个`anchor`被放置在滑动窗的中心。默认使用`3`个尺度和`3`个高宽比，所以**每一个滑动窗位置上有`9`个`anchor`**。对于一个`W×H`的卷积`feature map`而言，共有`WHk`个`ahchor`。
- 每一个滑动窗被映射为一个低维特征，其实就是一个`3x3`的卷积核，卷积后得到`1`个数值，`256`个通道在一个窗口卷积后就是`256-d`的特征，后面跟随`ReLu`激活函数。然后经过其他层，最后被输入进两个并行的全连接层:`box-regression layer` (`reg`) 和`box-classification layer` (`cls`)
<center><img src="/images/Faster-R-CNN/Region_proposal.png" width="60%"/></center>

本文的方法具有平移不变性，`anchor`和相对于`anchor`的`proposal`都是平移不变的。也就是说，如果图像中的目标被平移，`proposal`也应该平移，并且网络依然可以去预测任何位置的`proposal`。这种平移不变性通过`FCN`中的方法保证。平移不变性也减少了模型尺寸。

### Multi-Scale Anchors as Regression References
多尺度预测一般有两种方法：
- 基于图像/`feature`金字塔的:`SPPnet`,`Fast R-CNN`
- 在`feature map`上使用多尺度的滑动窗，如`DMP`

<center><img src="/images/Faster-R-CNN/multiple_scale.png" width="200%"/></center>

本文基于`anchor`金字塔，实现多尺度是根据多尺度和多个高宽比的`anchor`做分类和回归。这种方法只依赖单尺度的图像和`feature map`，并且只使用单尺度的`filter`（`feature map`上的滑动窗，就是前面说的`3x3`）。

**在`SPPnet`和`Fast R-CNN`中，`bounding box`回归是在从任意大小的`RoI`池化得到的特征上实现的，回归的权重被所有的`region`尺寸共享。`Faster R-CNN`针对不同尺寸，学习`k`个`bounding-box`回归器，每一个回归器只负责一个尺度和高宽比，`k`个回归器之间不共享权重**。在后面的`Details`里会讲到，使用了`36`个`1x1`的卷积实现回归的。因为这样的`anchor`设计，即使特征是固定尺寸或尺度的，也可能预测出不同尺寸的`box`。

### Loss Function
- 为训练`RPN`，对每一个`anchor`分配一个二分类标签：目标或非目标。
- 把一个正标签分配给两种`anchor`：与一个`ground-truth box`有着最高`IoU`的一个或多个`anchor`；与任一个`ground-truth box`的`IoU`大于`0.7`的一个`anchor`。注意一个`ground-truth box`可能给多个`anchor`分配了正的标签。（这里我的理解是：每一个`ground truth`对应的正样本`anchor`应该与它`IoU`最高的，但是同时也把那些与它`IoU`大于`0.7`的`anchor`也当做正的，因为涉及到训练深度神经网络需要样本量比较大，因此需要放松条件，不仅是非常符合要求的被选为正样本，也要考虑那些比较符合要求的）
- 负标签分配给那些与所有的`ground-truth box`的`IoU`都低于`0.3`的`anchor`。那些与真实值`IoU`为`[0,0.2]`的`anchor`不参与训练。

使用`multi-task loss`:

$$L({p_i},{t_i}) = \frac {1} {N_{cls}} \sum_i L_{cls}(p_i, p_i^\*) + \lambda \frac 1 {N_{reg}} \sum_i p_i^\* L_{reg}(t_i,t_i^\*)$$

带$p_i^\*$（值为1或0）是第`i`个`anchor`是否为目标的真实值，只有`anchor`为正样本时，回归的损失才会被算入。



### Training RPNs
- **一个`mini-batch`使用一张图像，在其中采样`256`个`anchor`**，正负`anchor`的比例是`1:1`，如果正的样本少于负的，则用负样本去填充这个`batch`，用这`256`个`anchor`计算损失
- 使用均值为`0`，标准差为为`0.01`的高斯分布随机初始化所有新添加的层，其他的共享卷积层使用`ImageNet`上预训练的模型初始化

## Sharing Features for RPN and Fast R-CNN
使用交替训练（`Alternating Training`）实现`RPN`和`Fast R-CNN`共享卷积特征，分为4步：
- 使用`ImageNet`预训练的模型初始化网络，并且为`region proposal`任务微调网络。
- 使用`RPN`网络生成的`proposal`，通过`Fast R-CNN`训练一个单独的检测网络。这个检测网络也是使用`ImageNet`预训练模型初始化的。到此时，两个网络并没有共享卷积层。
- 使用检测网络初始化`RPN`的训练，但是固定共享卷积层，只微调`RPN`特有的层，实现了两个网络共享卷积层。
- 固定共享卷积，微调`Fast R-CNN`特有的层。如此，两个网络共享了相同的卷积层，并且形成了一个统一的网络。

# Details
<center><img src="/images/Faster-R-CNN/network.png" width="100%"/></center>

- 输入图像经过`CNN`，这里是`VGG16`，得到`feature map`
- `feature map`首先输入给`RPN`，做了`3x3`的卷积，然后分两路：假如前一步`3x3`卷积后的特征尺寸是是`WxHxC`，在输入给`softmax`前使用了`18`个`1x1xC`的卷积，得到`WxHx18`的矩阵，然后分类每个`anchor`是不是目标；使用`36`个`1x1`的卷积，得到`WxHx36`的矩阵，相当于`feature map`上每一个位置都有`9`个`anchor`，这样得到回归后`proposal`的位置。
- `RoI Pooling`根据`RPN`的输出，从`feature map`里提取`proposal`对应的特征，并且池化成固定尺寸的输出
- 最后是全连接层，分类`proposal`是哪一类目标；回归`bounding box`

**Faster R-CNN进行了两次`bounding box`回归，一次是在`RPN`网络，针对`anchor`进行回归，目的是使`proposal`的位置更加接近真实值；一次是在全连接层之后，进行最后的位置回归**。

上图中有`4`个池化层，`VGG`的特点是每次都是在池化层改变`feature map`的尺寸。在`RPN`开始使用的`3x3`卷积也是保持`feature map`尺寸不变的，因此原始图像到`RPN`的`feature map`被缩放了`16`倍，所以要想将`proposal`映射会原图，只需要乘以一个缩放因子。

对于一幅`1000x600`的图像，经卷积后得到`60x40x9 = 20000`个`anchor`（`1000/16,600/16`）
- 在训练阶段，一幅图中会有约`6000`个`anchor`会超过图像的边界，这部分被去除掉；在测试阶段，对于越过边界的`anchor`，把它修正到边界上。
- 设置`0.7`的阈值进行`NMS`后，留下约`2000`的候选框
- 再对剩下的排序它们的得分，提取`top-N`个作为最终的`proposal`。文中经过试验选取前`300`个效果最好。

# Reference 
1. [论文原文](https://arxiv.org/abs/1506.01497)
2. [faster-RCNN算法原理详解](https://blog.csdn.net/wfei101/article/details/76400672)

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
