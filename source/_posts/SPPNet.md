---
title: SPPNet:Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
date: 2018-02-18 20:17:12
categories: CNN
tags: 
  - object detection
  - paper
image: /images/SPPNet/spp.png
---
> 为了解决现有`CNN`需要固定输入大小的问题，提出了`SPP-net`，使得针对任意尺寸的图像生成固定长度的特征表示。输入一张图，只需要对整张图进行一次`feature map`的计算，避免了像`R-CNN`那样重复地计算卷积特征。`SPP-net`不仅可以应用在分类任务上，而且在检测任务上也有很大的性能提升。
<!-- more -->

# Introduction
在`CNN`的训练和测试阶段都有一个技术问题：`CNN`需要固定输入图像的尺寸，这些图片或者经过裁切（`crop`）或者经过变形缩放（`warp`），都在一定程度上导致图片信息的丢失和变形，限制了识别精确度。

如下图所示，上面是`CNN`一般的做法，对不符合网络输入大小的图像直接进行`crop`或`warp`，下面是`SPP-net`的工作方式。`SPP-net`加在最后一个卷积层的输出后面，使得不同输入尺寸的图像在经过前面的卷积池化过程后，再经过`SPP-net`，得到相同大小的`feature map`，最后再经过全连接层进行分类。

<center><img src="/images/SPPNet/structure.png" width="70%"/></center>

# Spatital Pyramid Pooling
`CNN`为什么需要固定输入尺寸？**卷积层是不需要输入固定大小的图片，而且还可以生成任意大小的特征图，只是全连接层需要固定大小的输入。因此，固定输入大小约束仅来源于全连接层**。在本文中提出了`Spatial Pyramid Pooling layer`来解决这一问题，输入任意尺寸的图像，`SPP layer`对特征进行池化并生成固定大小的输出，以输入给全连接层或分类器。

- 以`AlexNet`为例，经`CNN`得到`Conv5`输出的任意尺寸的`Feature map`，图中`256`是`conv5`卷积核的数量。
- 将最后一个池化层`pool5`替代成`SPP layer`。以不同网格来提取特征，分别是`4x4`，`2x2`，`1x1`，将这三张网格放到`feature map`上，就可以得到`16+4+1=21`种不同的块(`Spatial bins`)，对这`21个`块应用`max pooling`，每个块就提取出一个特征值，这样就组成了`21`维特征向量。

这种**用不同大小的格子划分`feature map`，然后对每一个块应用最大池化，将池化后的特征拼接成一个固定维度的特征的方式就是空间金字塔池化**。

<center><img src="/images/SPPNet/spp.png" width="70%"/></center>

总结而言，当网络输入一张任意大小的图片，进行卷积、池化，直到即将与全连接层连接的时候，就要使用空间金字塔池化，使得任意大小的特征图都能够转换成固定大小的特征向量，这就是空间金字塔池化的意义（**不同度特征提取出固定大小的特征向量**）。

# Training
理论上，无论输入什么尺寸的图像，都可以用标准的反向传播训练。但是实际上由于`GPU`实现中，更适合在固定尺寸的输入图像上，因此提出了一些训练策略。
- `Single-size training`:使用固定的`224x224`的输入，是从原始图像中裁切得到的，目的是为了数据扩增；对于给定的输入尺寸，可以预先计算出空间金字塔池化需要的`bin size`，假如`feature map`是`axa`的大小，那么在`SPP layer`中，窗口尺寸$win = \frac a n$，步长$stride = \frac a n$。
- `Multi-size training`：考虑两种输入，`180x180`和`224x224`，这里不再用裁切，而是直接进行缩放，比如把`224x224`的图像直接缩放为`180x180`，它们之间的区别只是分辨率不同。实现两个固定输入尺寸的网络，训练过程中先在`1`号网络上训练一个`epoch`，然后用它的权重去初始化`2`号网络，训练下一个`epoch`；如此转换训练。通过共享两种尺寸输入的网络参数，实现了不同输入尺寸的`SPP-Net`的训练。

这样`single/multi-size`的训练只是在训练中，预测阶段，直接将不同尺寸的图像输入给`SPP-Net`。


# SPP-net for Object Detection
对于`R-CNN`，整个过程是：
- 首先通过选择性搜索，对待检测的图片进行搜索出约`2000`个候选窗口。 
- 把这`2000`个候选框都缩放到`227x227`，然后分别输入`CNN`中，利用CNN对每个`proposal`进行提取特征向量。 
- 把上面每个候选窗口的对应特征向量，利用`SVM`算法进行分类识别。 

可以看出**`R-CNN`的计算量是非常大的，因为`2k`个候选窗口都要输入到`CNN`中，分别进行特征提取**。

而对于`SPP-Net`，整个过程是：
- 首先通过选择性搜索，对待检测的图片生成`2000`个候选窗口。这一步和`R-CNN`一样。
- 特征提取阶段。与`R-CNN`不同，把整张待检测的图片，输入`CNN`中，进行一次特征提取，得到整个图像的`feature maps`（可能是在多尺度下），然后在`feature maps`中找到各个候选框对应的区域，对各个候选框采用空间金字塔池化，提取出固定长度的特征向量。**因为`SPP-Net`只需要对整张图片进行一次特征提取，速度会大大提升**。文中是几十到一百倍以上。
- 最后一步也是和`R-CNN`一样，采用`SVM`算法进行特征向量分类识别。

<center><img src="/images/SPPNet/compare.png" width="70%"/></center>

下图为`SPP-net`进行目标检测的完整步骤：
<center><img src="/images/SPPNet/whole.png" width="70%"/></center>

# Mapping a Window to Feature Maps
`SPP-Net`在提取完整图像的`feature map`后，要将候选框的位置映射到`feature map`中得到对应特征。候选框是在原始图像上得到的，而`feature maps`是经过原始图片卷积、下采样等一系列操作后得到的，所以`feature maps`的大小和原始图片的大小是不同的。

假设$(x,y)$是原始图像上的坐标点，$(x',y')$是特征图上的坐标，`S`是`CNN`中所有的步长的乘积，那么左上角的点转换公式如下：$$x' = \frac x S + 1$$
右下角的点转换公式为：$$x' = \frac x S - 1$$
# Reference
1. [论文原文](https://arxiv.org/pdf/1406.4729.pdf)
2. [SPPnet论文总结](http://blog.csdn.net/xjz18298268521/article/details/52681966)
3. [SPP-Net论文详解](http://blog.csdn.net/v1_vivian/article/details/73275259)

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
