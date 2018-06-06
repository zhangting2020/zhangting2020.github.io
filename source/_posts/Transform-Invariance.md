---
title: 卷积神经网络为什么具有平移不变性？
date: 2018-05-30 22:17:22
categories: CNN
tags: CNN
---
> 在我们读计算机视觉的相关论文时，经常会看到平移不变性这个词，本文将介绍卷积神经网络中的平移不变性是什么，以及为什么具有平移不变性。
<!-- more -->
# 什么是平移不变性

## 不变性

不变性意味着即使目标的外观发生了某种变化，但是你依然可以把它识别出来。这对图像分类来说是一种很好的特性，因为我们希望图像中目标无论是被平移，被旋转，还是被缩放，甚至是不同的光照条件、视角，都可以被成功地识别出来。

所以上面的描述就对应着各种不变性：

- 平移不变性：Translation Invariance
- 旋转/视角不变性：Ratation/Viewpoint Invariance
- 尺度不变性：Size Invariance
- 光照不变性：Illumination Invariance

## 平移不变性/平移同变性

在欧几里得几何中，平移是一种几何变换，表示把一幅图像或一个空间中的每一个点在相同方向移动相同距离。比如对图像分类任务来说，图像中的目标不管被移动到图片的哪个位置，得到的结果（标签）应该是相同的，这就是卷积神经网络中的平移不变性。

**平移不变性意味着系统产生完全相同的响应（输出），不管它的输入是如何平移的 。平移同变性（translation equivariance）意味着系统在不同位置的工作原理相同，但它的响应随着目标位置的变化而变化** 。比如，实例分割任务，就需要平移同变性，目标如果被平移了，那么输出的实例掩码也应该相应地变化。最近看的FCIS这篇文章中提到，一个像素在某一个实例中可能是前景，但是在相邻的一个实例中可能就是背景了，也就是说，同一个像素在不同的相对位置，具有不同的语义，对应着不同的响应，这说的也是平移同变性。

# 为什么卷积神经网络具有平移不变性

简单地说，卷积+最大池化约等于平移不变性。

- 卷积：简单地说，图像经过平移，相应的特征图上的表达也是平移的。下图只是一个为了说明这个问题的例子。输入图像的左下角有一个人脸，经过卷积，人脸的特征（眼睛，鼻子）也位于特征图的左下角。

  <center><img src="/images/平移不变性/fig1.png" width=“40%”></center>

  假如人脸特征在图像的左上角，那么卷积后对应的特征也在特征图的左上角。

  <center><img src="/images/平移不变性/fig2.png" width=“40%”></center>

  在神经网络中，卷积被定义为不同位置的特征检测器，也就意味着，无论目标出现在图像中的哪个位置，它都会检测到同样的这些特征，输出同样的响应。比如人脸被移动到了图像左下角，卷积核直到移动到左下角的位置才会检测到它的特征。

- 池化：比如最大池化，它返回感受野中的最大值，如果最大值被移动了，但是仍然在这个感受野中，那么池化层也仍然会输出相同的最大值。这就有点平移不变的意思了。

 所以这两种操作共同提供了一些平移不变性，即使图像被平移，卷积保证仍然能检测到它的特征，池化则尽可能地保持一致的表达。

# Reference

1. [How is a convolutional neural network able to learn invariant features? ](https://www.quora.com/How-is-a-convolutional-neural-network-able-to-learn-invariant-features)
2. [Why and how are convolutional neural networks translation-invariant? ](https://www.quora.com/Why-and-how-are-convolutional-neural-networks-translation-invariant)

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
