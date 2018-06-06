---
title: Fast-R-CNN
date: 2018-02-28 17:48:31
categories: CNN
tags: 
  - object detection
  - paper
image: /images/Fast-R-CNN/framework.png
---
> 这篇文章发表在`ICCV2015`上，为了改进`R-CNN`，`SPPnet`多阶段训练的缺点，以及`SPPnet`限制了误差的反向传播的缺点，提出了`Fast R-CNN`。在训练过程中，使用`multi-task loss`简化了学习过程并且提高了检测准确率。
<!-- more -->

# Introduction

**`RCNN`的三大缺点：**
- 多阶段训练：首先用交叉熵损失微调卷积神经网络；然后线性`SVM`拟合卷积特征；最后学习`bounding-box`回归
- 训练代价高（空间及时间）：从每幅图中的每个`region proposal`提取的特征需要存储起来
- 测试慢

`R-CNN`之所以慢，就是因为它独立地`warp`然后处理每一个目标`proposal`。流程如下：

提取`proposal` -> `CNN`提取`feature map` -> `SVM`分类器 -> `bbox`回归

**`SPPnet`的提出是为了加速`R-CNN`。但是具有以下缺点：**
- 同`R-CNN`一样，多阶段，特征需要被写入磁盘
- 不同于`R-CNN`的是：微调算法只更新那些跟随在`SPP layer`后的全连接层。

**Fast R-CNN的贡献**
- 比`R-CNN`更高的检测质量（`mAP`）
- 训练时单阶段的，使用`multi-task loss`
- 在训练过程中，所有的网络层都可以更新
- 不需要对特征存入磁盘
`R-CNN`，`SPPNet`在检测器的训练上都是多阶段的训练，训练起来困难并且消耗时间。`SPP-Net`限制了训练过程中误差的反向传播，潜在地限制了精确度；目标候选位置需要被精修，过去的精修是在一个单独的学习过程中训练的，`Fast-RCNN`是对检测器的训练是单阶段的。

# Fast R-CNN Training
网络结构上：卷积+池化层 -> `RoI pooling layer` -> 全连接层。两个并行的层：一个输出类别概率，一个输出四个实值即`bounding box`。
<center><img src="/images/Fast-R-CNN/architecture.png" width="70%"/></center>

## RoI pooling layer
**`RoI pooling layer`是`SPPnet`中`SPP layer`的简化版本，相当于金字塔只有一级**。`SPP-Net`中设置了不同样子的网格，比如`4x4`，`2x2`，`1x1`的。
- `RoI pooling layer`的输入是`N`个`feature map`和`R`个感兴趣的区域构成的列表， `R>>N`
- `N`个`feature map`是由网络的最后一个卷积层提供的，并且每一个都是多维矩阵`H×W×C`。
- 每一个`RoI`是一个元组`（n,r,c,h,w）`，指定了`feature map`的索引`n（n为0~N-1）`和`RoI`的左上角位置`（r,c）`以及高和宽`（h,w）`。
- `RoI pooling`层输出`H'× W'`的`feature map`，通道数和原始的`feature map`一样（其中，H' <= H, W' <=W）。

`RoI pooling`的具体操作如下：
- 首先将`RoI`映射到`feature map`对应的位置
- 将映射后的区域划分为一定大小的块（`bin`），尺寸为`h/H' × w/W'`，`h`是`feature map`中`ROI`的高，`H'`是要求输出的`feature map`的高。
- 对每一个块进行`max pooling`操作

如下图，输入8×8的`feature map`，一个`RoI`（黑色的大框），希望的输出是2×2的。
<center><img src="/images/Fast-R-CNN/RoI_pooling.png" width= "40%"/></center>

首先找到`RoI`在`feature map`中的位置，其大小为`7x5`；映射后的区域划分为3×2（7/2=3,5/2=2）的块，可能出现如图中，不能整除的情况；最后对每一个块单独进行`max pooling`，得到要求尺寸的输出。
<center><img src="/images/Fast-R-CNN/RoI_pooling_result.png" width= "20%"/></center>

整个检测框架表示为：
<center><img src="/images/Fast-R-CNN/framework.png" width= "70%"/></center>

**总结一下，`Fast R-CNN`先用基础网络提取完整图像的`feature map`，将`selective search`提取的候选框作为`RoI`，把`feature map`和`RoI`输入给`RoI pooling layer`，在`feature map`中找到每一个`RoI`的位置，根据需要的输出尺寸，把那部分`feature map`划分网格，对每一个网格应用最大池化，就得到了固定尺寸的输出特征**。

## Using pretrained networks
使用预训练的网络初始化`Fast R-CNN`，要经历三个转变：
- 最后一个最大池化层使用`RoI`池化层替代。通过设置`RoI pooling layer`的输出尺寸`H'`和`W'`与网络第一个全连接层兼容。
- 网络的最后一个全连接层和`softmax`被替代为两个并行的层。
- 网络采取两个数据输入：`batch size`为`N`的输入图像和`R`个`RoIs`的列表。

`SPP-Net`最后是一个`3`层的`softmax`分类器用于检测（`SPP layer`后面是两个全连接层，和一个输出层）。由于卷积特征是离线计算的，所以微调过程不能向`SPP layer`以下的层反向传播误差。以`VGG16`为例，前`13`层固定在初始化的值，只有最后`3`层会被更新。

在`Fast R-CNN`中，`mini-batch`被分层次地采样，首先采样图像，然后采样这些图像的`RoIs`。来自同一幅图的`RoI`共享计算和内存，使得训练高效。

## Multi-task loss
`Fast R-CNN`是并行地进行类别的确定和位置的精修的，整体的`loss`由两部分组成，一部分是分类的损失，另一部分是位置回归的损失，因此定义的损失如下，$k^\*$为真实的类别标签，$[k^\* \ge 1]$表明只对目标类别计算损失，背景类别的$k^\*=0$，提取出的`RoI`是背景的话，就忽略掉：
$$L(p,k^\*,t,t\*) = L_{cls}(p,k^\*) + \lambda [k^\* \ge 1] L_{loc}(t, t^\*)$$ 

$$L_{loc}(t,t^\*) = \sum_{i \in {x,y,w,h}} smooth_{L1}(t_i,t^\*_i)$$ 

$$smooth_{L1}(x)=\begin{cases} 
		0.5x^2, & if~|x|<1 \\ 
		|x|-0.5, & otherwise
	\end{cases}$$


对于`bounding box`回归使用`Smooth L1 loss`是因为，比起`R-CNN`中使用的`L2 loss`，`Smooth L1 loss`对于离群值不敏感。归一化了`ground truth`的回归目标$t^*$使其具有`0`均值和单位方差，这样的情况下设置$\lambda = 1$在实验中效果很好。


## Detail
- 微调中，`batch size N = 2`，`R=128`，也就是每一幅图采样了`64`个`RoI`
- `N`张完整图片以`50%`概率水平翻转
- `R`个候选框的构成：与某个真实值`IoU`在`[0.5,1]`的候选框被选为`RoIs`；与真实值`IoU`在`[0.1,0.5]`的候选框作为背景， 标记类别为$k^*=0$
- 多尺度训练中，和`SPP-Net`一样，随机采样一个尺度，每一次采样一幅图

# Reference
1. [论文原文](https://arxiv.org/abs/1504.08083)
2. [ROI Pooling层详解](http://blog.csdn.net/auto1993/article/details/78514071)
3. [Training R-CNNs of various velocities](http://mp7.watson.ibm.com/ICCV2015/slides/iccv15_tutorial_training_rbg.pdf)

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
