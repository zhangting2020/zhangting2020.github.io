---
title: R-CNN:Rich feature hierarchies for accurate object detection and semantic segmentation
date: 2018-02-17 13:52:18
categories: CNN
tags: 
  - object detection
  - paper
image: /images/R-CNN/overview.png
---
> 这篇文章发表在`CVPR2014`上面，是第一篇展示`CNN`可以在`PASCAL VOC`数据集上带来明显更好的目标检测性能的文章。提出了一种使用`selective search` + `CNN`的两阶段的目标检测方法。创新点在于使用`CNN`提取特征，在大数据集下有监督的预训练，小数据集上微调解决样本数量少难以训练的问题。
<!-- more -->

# Introduction
使用`CNN`做目标检测需要解决两个问题：一是使用深度网络定位目标；二是使用少量标记数据训练一个高性能的模型。与图像分类不同，目标检测是要在一幅图像中检测出目标，可能有多个。一种方法是将其作为回归问题，比如构建一个滑动窗检测器。

本文提出的`R-CNN`的整体框架如图，也因为将`region`和`CNN feature`组合，所以作者定义为`R-CNN（region with CNN features）`。
- 输入一幅图像，产生约`2000`个类别独立的`region proposal`
- 然后使用仿射图像扭曲（`affine image warping`）将每一个`region proposal`转换为固定尺寸的`CNN`输入
- 再使用`CNN`从每一个`proposal`中提取固定长度的特征向量
- 对每一个类别训练一个`SVM`，然后使用这些特定类别的线性`SVM`对每一个区域进行分类。
<center><img src="/images/R-CNN/overview.png" width="70%"/></center>

传统的方法解决数据稀缺问题时，使用无监督的预训练和有监督的微调。这篇文章第二个贡献就是当数据稀缺时，在更大辅助数据集(`ILSVRC`)上使用监督式预训练，然后在小的数据集（`PASCAL`）上进行特定域的微调，是一种学习高性能的`CNN`的有效范例。简而言之，就是**使用图像分类中的经典网络作为基础网络，然后在目标检测这种任务上，进行微调**。

# Object detection with R-CNN
分为`3`个模块：
- 第一个模块生成类别独立的`region proposals`，使用`selective search`生成。就是采取过分割手段，将图像分割成小区域，再通过颜色直方图，梯度直方图相近等规则进行合并，最后生成约`2000`个候选框
- 第二个模块是一个大的卷积神经网络，从每一个区域中提取固定长度的特征向量。通过前向传播`227x227`的图像（减均值），经过`5`个卷积层和`2`个全连接层，对每一个`region proposal`计算得到`4096`的特征向量
- 第三个模块是一系列的特定类别的线性`SVM`。

## Test
- 选择性搜索提取候选区域框，每个候选框周围加上`16`个像素值为候选框像素平均值的边框，再直接缩放到网络输入`227x227`
- 每一个候选框输入到`CNN`之前，先减去均值，经`AlexNet`网络提取`4096`维的特征，`2000`个候选框就组成`2000x4096`的矩阵
- 将`2000×4096`维特征与`20`个`SVM`组成的权值矩阵`4096×20`相乘，获得每个候选框对应类别的得分，`Pascal VOC`数据集有`20`类目标
- 每一类都有多个候选框，因此要进行非极大值抑制，去掉重叠候选框
- 用`20`个线性回归器对得到的每个类别的候选框进行回归，获取目标位置

## Training
- 监督式预训练，在`ImageNet`数据集上进行，使用图像分类的数据，只有类别标签，没有`bounding box`。
- 特定域的微调
  - 只使用来源于`VOC`的数据做随机梯度下降，以训练`CNN`参数，初始学习率设为`0.001`，每一次`SGD`迭代(这里指的就是`mini-batch`梯度下降)，在所有类别中均匀采样`32`个正样本，和`96`个背景区域，组成一个`128`的`mini-batch`
  - 用`21way`的分类层替代原来的`1000way`分类层。
  - 根据`bounding box`的所属的类别，把那些与`ground truth`的`IoU`大于`0.5`的`region proposal`作为正样本，其余的作为负样本。
- 目标分类器
  - 一个图像区域紧紧包围住一辆车，则这个区域就是正样本。所以正样本就是`bounding box`的`ground truth`。文中选择`0.3`作为`IoU`阈值，低于这个阈值的区域作为负样本。
  - 提取好样本的特征，就可以优化每一个类别的`SVM`。由于训练数据太多，采用标准的难负样本挖掘方法（`hard negative mining `），可以使训练快速收敛。

这里也可以看出，在微调`CNN`和训练`SVM`时，对于正负样本`IoU`阈值的限定不一样，前者的限定更加宽松，这是因为`CNN`需要大量的样本，否则会过拟合，而`SVM`就可以使用相对少量的样本，故限制更严格。

## Detail
- `bounding box`回归：选择性搜索产生的`region proposal`输入到`CNN`中，文章中`CNN`使用的是`AlexNet`，将`AlexNet`的`Pool5`产生的特征用来训练线性回归模型，从而预测`bounding box`的位置。在回归中，如果候选框与`ground truth`距离太远，训练是很困难的几乎没有希望，所以在样本对的选择上，只选择那些与`ground truth`离得近的，文中通过设置`proposal`和`IoU`阈值为`0.6`，低于阈值的`proposal`就没有匹配的`ground truth`，被忽略掉。
- 目标类别：使用倒数第二个全连接层`fc7`输出的特征，训练分类器`SVM`或者`softmax`
- 两者的结合：一个`region proposal`送入卷积神经网络，`Pool5`和倒数第二个全连接层`fc7`的特征会先保存下来；然后使用所有类别的`SVM`对这个`proposal`预测类别分数，决定是否为某种目标；如果是某种目标，则用相应类别的线性回归去预测得到`bounding box`的位置。

# 问题
- 处理速度慢，主要是因为一张图片产生的约`2k`个候选框由`CNN`提取特征时，有很多区域的重复计算
- 整个测试过程也比较繁琐，要经过两阶段，而且单独进行分类和回归，这些不连续的过程在训练和测试中必然会涉及到特征的存储，因为会浪费磁盘空间。

# Reference
1. [论文原文](https://arxiv.org/abs/1311.2524)
2. [R-CNN论文详解](https://www.cnblogs.com/zf-blog/p/6740736.html)

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
