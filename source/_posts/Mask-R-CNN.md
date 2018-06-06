---
title: Mask-R-CNN
date: 2018-05-06 21:47:07
categories: CNN
tags:
  - instance segmentation
  - object detection
  - paper
image: /images/Mask-R-CNN/framework.png
---

> 这篇论文提出了一种概念简单，灵活且通用的目标实例分割框架，在检测出图像中目标的同时，生成每一个实例的掩码（`mask`）。对`Faster R-CNN`进行扩展，通过添加与已存在的`bounding box`回归平行的一个分支，预测目标掩码，因而称为`Mask R-CNN`。这种框架训练简单，容易应用到其他任务，比如目标检测，人体关键点检测。
<!-- more -->

# Introduction
实例分割的挑战性在于要求正确地检测出图像中的所有目标，同时精确地分割每一个实例。这其中包含两点内容：
- 目标检测：检测出目标的`bounding box`，并且给出所属类别；
- 语义分割（`semantic segmentation`）：分类每一个像素到一个固定集合，不用区分实例。

`Mask R-CNN`对`Faster R-CNN`进行了扩展，在`Faster R-CNN`分类和回归分支的基础上，添加了一个分支网络去预测每一个`RoI`的分割掩码，把这个分支称为掩码分支。掩码分支是应用在每一个`RoI`上的一个小的`FCN`，以像素到像素的方式（pixel-to-pixel）预测分割掩码。

<center><img src="/images/Mask-R-CNN/framework.png" width="50%"/></center>

`Faster R-CNN`在网络的输入和输出之间没有设计像素到像素的对齐。在`how RoIPool`文中提到：实际上，应用到目标上的核心操作执行的是粗略的空间量化特征提取。为了修正错位，本文提出了`RoIAlign`，可以保留准确的空间位置，这个改变使得掩码的准确率相对提高了`10%`到`50%`。解耦掩码和分类也至关重要，本文对每个类别独立地预测二值掩码，这样不会跨类别竞争，同时依赖于网络的`RoI`分类分支去预测类别。

模型在`GPU`上运行每帧`200ms`，在`8 GPU`的机器上训练`COCO`数据集花费了一到两天。最后，通过`COCO`关键点数据集上的人体姿态估计任务来展示框架的通用性。通过将每个关键点视为一位有效编码（`one-hot`），即所有关键点编码成一个序列，但只有一个是`1`，其余都是`0`。只需要很少的修改，`Mask R-CNN`可以应用于人体关键点检测。不需要额外的技巧，`Mask R-CNN`超过了`COCO 2016`人体关键点检测比赛的冠军，同时运行速度可达`5FPS`。

# Related Work
早前的实例分割方法受`R-CNN`有效性的推动，基于分割`proposal`，也就是先提取分割候选区，然后进行分类，分割先于分类的执行。本文的方法是同时预测掩码和类别，更加简单和灵活。

`FCIS`（`fully convolutional instance segmentation`）用全卷积预测一系列位置敏感的输出通道，这些通道同时处理目标分类，目标检测和掩码，这使系统速度变得更快。但`FCIS`在重叠实例上出现系统错误，并产生虚假边缘。

另一类方法受语义分割的推动，将同类别的像素划分到不同实例中，这是一种分割先行的策略。`Mask R-CNN`与其相反，基于实例先行的策略（`segmentation-first strategy`）。

# Mask R-CNN
`Mask R-CNN`在`Faster R-CNN`上加了一个分支，因此有三个输出：目标类别、`bounding box`、目标掩码。但是掩码输出与其他输出不同，需要提取目标更精细的空间布局。`Mask R-CNN`中关键的部分是像素到像素的对齐，这在`Fast/Faster R-CNN`里是缺失的。

首先回归一下`Faster R-CNN`：它包含两个阶段，第一阶段使用`RPN`提取候选的目标`bounding box`，第二阶段本质上是`Fast R-CNN`，使用`RoI pooling`从候选区域中提取特征，实现分类并得到最终的`bounding box`。

`Mask R-CNN`也是两个阶段：第一阶段与`Faster R-CNN`相同，`RPN`提取候选目标`bounding box`；第二阶段，除了并行地预测类别和候选框偏移，还输出每一个`RoI`的二值掩码（`binary mask`）。

## 损失函数
- 多任务损失：$$L=L_{cls}+L_{box}+L_{mask}$$ 掩码分支对每一个感兴趣区域产生$Km^2$维的输出，`K`是类别数目，`K`个分辨率为`m×m`的二值掩码也就是针对每一个类别产生了一个掩码。
- 对每一个像素应用`sigmoid`，所以掩码损失就是平均二分类交叉熵损失。如果一个`RoI`对应的`ground truth`是第`k`类，那么计算掩码损失时，只考虑第`k`个掩码，其他类的掩码对损失没有贡献。
- 掩码损失的定义允许网络为每个类别独立预测二值掩码。使用专门的分类分支去预测类别标签，类别标签用来选择输出掩码。

## 掩码表达
- 掩码编码了输入目标的空间布局。掩码的空间结构，可以通过卷积产生的那种像素到像素的对应关系来提取。
- 使用`FCN`为每个`RoI`预测一个`m×m`的掩码。这允许掩码分支中的每个层显式的保持`m×m`的目标空间布局，而不会将其缩成缺少空间维度的向量表示。
- 像素到像素的对应需要`RoI`特征（它们本身就是小特征图）被很好地对齐，以准确地保留显式的像素空间对应关系。

## RoI Align
首先说明为什么需要对齐，下图中左边是`ground truth`，右边是对左边的完全模仿，需要保持位置和尺度都一致。平移同变性（`translation equivariance`）就是输入的改变要使输出也响应这种变化。
- 分类要求平移不变的表达，无论目标位置在图中如何改变，输出都是那个标签
- 实例分割要求同变性：具体的来说，就是平移了目标，就要平移掩码；缩放了目标就要缩放掩码

**全卷积网络`FCN`具有平移同变性，而卷积神经网络中由于全连接层或者全局池化层，会导致平移不变**。
<center><img src="/images/Mask-R-CNN/translation.png" width="50%"/></center>

在`Faster R-CNN`中，提取一张完整图像的`feature map`，输入`RPN`里提取`proposal`，在进行`RoI pooling`前，要根据`RPN`给出的`proposal`信息在基础网络提取出的整个`feature map`上找到每个`proposal`对应的那一块`feature map`，具体的做法是：根据`RPN`给出的边框回归坐标，除以尺度因子`16`，因为`vgg16`基础网络四次池化缩放了`16`倍。这里必然会造成坐标计算会出现浮点数，而`Faster R-CNN`里对这个是进行了舍入，这是一次对平移同变性的破坏；同样的问题出现在后面的`RoI pooling`中，因为要得到固定尺寸的输出，所以对`RoI`对应的那块`feature map`划分了网格，也会出现划分时，对宽高做除法出现浮点数，这里和前面一样，简单粗暴地进行了舍入操作，这是第二次对平移同变性的破坏。如下图，网格的划分是不均匀的：
<center><img src="/images/Mask-R-CNN/RoIPool.png" width="60%"/></center>
<center><img src="/images/Mask-R-CNN/RoIPooling.png" width="60%"/></center>

总之，`Faster R-CNN`破坏了像素到像素之间的这种平移同变性。`RoI Align`就是要在`RoI`之前和之后保持这种平移同变性，避免对`RoI`边界和里面的网格做量化。如下图：
- 针对输入的`feature map`找到对应的`RoI`，是通过$x/16$而不是像`Faster R-CNN`中$[x/16]$，$[\cdot]$代表舍入操作。所以可以看到第一幅图中`RoI`并没有落在整数的坐标上。
- 对`RoI`划分为`2x2`的网格（根据输出要求），每个小的网格里采样`4`个点，使用双线性插值根据临近的网格点计算这`4`个点的值，最后再对每一个网格进行最大池化或平均池化得到最终`2x2`的输出。
<center><img src="/images/Mask-R-CNN/RoIAlign2.png" width="60%"/></center>
<center><img src="/images/Mask-R-CNN/RoIAlign.png" width="90%"/></center>

## network
下图中，是两个不同的`network head`，左图是`ResNet C4`，右边是`FPN`主干，这两种结构上都添加了一个掩码分支，反卷积使用`2x2`的卷积核，`stride`为`2`；除了输出层是`1x1`的卷积，其他部分的卷积都是`3x3`的。
<center><img src="/images/Mask-R-CNN/network.png" width="70%"/></center>

## 实现细节
- 掩码损失只定义在正的`RoI`上
- 输入图像被缩放到短边为`800`，每个图像采样`N`个`RoI`（`ResNet`的`N=64`，`FPN`的`N=512`），`batch size = 2`，正负样本的比例为`1:3`。
- 测试中对于`ResNet`架构，生成`300`个`proposal`，`FPN`则是`1000`。
- 将得分最高的`100`个检测框输入掩码分支，对每一个`RoI`预测出`K`个掩码，但是最终只根据分类分支的预测结果选择相应的那一个类别的掩码。
- `mxm`的浮点数掩码输出随后被缩放到`RoI`尺寸，然后以`0.5`的阈值进行二值化。

# Reference
1. [Mask R-cNN](https://arxiv.org/abs/1703.06870)

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

