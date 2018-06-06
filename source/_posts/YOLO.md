---
title: YOLO
date: 2018-04-13 14:26:52
categories: CNN
tags: 
  - object detection
  - paper
image: /images/YOLO/model.png
---
> 过去的目标检测都是用分类器去实现检测，本文构造的检测器，将目标检测作为一种回归空间分离的边界框和相关类概率的问题。一次预测中，一个神经网络（`single network`）直接从整幅图像中预测出边界框和类别概率，是一种端到端的实现。
<!-- more -->

# Introduction
当前的检测系统是把检测作为分类问题。系统使用目标分类器，并且在图像中不同位置和尺度上进行判断。如`R-CNN`先提出候选框，然后对每个候选框使用分类器，分类之后精修边界框并且去除重复的框。这种方式速度慢并且很难优化，因为每一个独立的部分都必须单独训练。

**本文将目标检测重新设计为一种单一的回归问题，直接从图像像素得到边界框坐标和类别概率**。在图像中只需要看一次（`You Only Look Once`）就知道目标是否存在，存在于哪里。

`YOLO`的大致流程如下图：一个单卷积网络同时预测多个`bounding box`和相应的类别概率。首先对输入图像缩放到`448×448`，然后对其运行单卷积网络，最后使用非极大值抑制消除重复框。

<center><img src="/images/YOLO/overview.png" width="70%"/></center>

# Unified Detection
- `YOLO`中，输入图像被划分为`S×S`的网格，如果一个目标的中心落在某一个网格中，则该网格负责检测这个目标。
- **每一个网格预测出`B`个`bounding box`以及这些`box`对应的置信度分数**，这些分数反映了模型有多大的把握认为这个`box`包含一个目标并且预测的`bounding box`有多精确。`Bounding box`信息包含`5`个数据值，分别是`x`,`y`,`w`,`h`,和`confidence`。其中`x`,`y`是指当前格子预测的物体的`bounding box`中心位置的坐标。`w`,`h`是`bounding box`的宽度和高度。`confidence = Pr(Object)*IoU`，如果该网格没有目标，则`Pr(Object)`为`0`，否则为`1`。所以当网格里有目标时，`condifence`应该与预测值和`ground truth`的`IoU`相等。
- 每一个网格也预测`C`个类别条件概率$Pr(Class_i|Obeject)$，表示在该网格包含目标的前提下，目标是某种类别的概率。不管一个网格预测多少个`bounding box`，总之对这个网格预测出`C`个类别条件概率。**`confidence`是针对每个`bounding box`的，而类别条件概率是针对每个网格的**。
<center><img src="/images/YOLO/box.png" width="58%"/></center>
<center><img src="/images/YOLO/score.png" width="60%"/></center>

上面中，组合每个网格预测的检测框和类条件概率，不仅得到了每个候选框的位置还得到了对应的类别概率。最后使用`NMS`消除重叠的框。

**总结一下`SSD`的思想**：将输入图像划分为`SxS`的网格，对每一个网格，预测`B`个`bounding box`，每个`bounding box`包含`4`个位置信息和`1`个`bounding box`置信度分数；同时对每一个网格还预测了`C`个类别条件概率。那么对一幅图，就会得到`SxS(Bx5+C)`的`tensor`。

作者在`VOC`数据上使用的是`S=7`，`B=2`，`C=20`，也就是最终得到一个`7x7x30`的`tensor`。

<center><img src="img/tensor1.gif" width="100%"/></center>


## Network Design
检测网络一共有`24`个卷积层和`2`个全连接层。其中可以看到`1×1`的降维层和`3×3`的卷积层的组合使用。

<center><img src="/images/YOLO/network.png" width="80%"/></center>

## Training
YOLO模型训练分为两步：
- 预训练。使用`ImageNet`中`1000`类数据训练`YOLO`网络的前`20`个卷积层 `+` `1`个`average`池化层 `+` `1`个全连接层。训练图像分辨率`resize`到`224x224`。
- 回到前面的网络图，加入`4`个卷积层和`2`个全连接层，构成`YOLO`网络。用上一步骤得到的前`20`个卷积层网络参数来初始化`YOLO`模型前`20`个卷积层的网络参数，然后用`VOC`中`20`类标注数据进行`YOLO`模型训练。检测要求细粒度的视觉信息，在训练检测模型时，将输入图像分辨率`resize`到`448x448`。

训练时，每个目标被匹配到对应的某个网格，训练过程中调整这个网格的类别概率，使真实目标类别的概率最高，其它的尽可能小，在每个网格预测的`bounding box`中找到最好的那个，并且调整它的位置，提高置信度，同时降低其它候选框的置信度。对于没有匹配任何目标的网格，降低这些网格中候选框的置信度，不用调整候选框位置和类别概率。
<center><img src="/images/YOLO/train.png" width="100%"/></center>
## Loss
`YOLO`使用平方和误差作为`loss`函数来优化模型参数。
- 位置误差（坐标、`IoU`）与分类误差对网络`loss`的贡献值是不同的，因此`YOLO`在计算`loss`时，使用权重为`5`的因子来修正位置误差。
- 在计算`IoU`误差时，包含物体的格子与不包含物体的格子，二者的`IOU`误差对网络`loss`的贡献是不同的。若采用相同的权值，那么不包含物体的格子的`confidence`值近似为`0`，变相放大了包含物体的格子的`confidence`误差在计算网络参数梯度时的影响。为解决这个问题，`YOLO` 使用权重为`0.5`的因子修正`IoU`误差。（注此处的“包含”是指存在一个物体，它的中心坐标落入到格子内）。
- 对于相等的误差值，大物体误差对检测的影响应小于小物体误差对检测的影响。这是因为，计算位置偏差时，大的`bounding box`上的误差和小的`bounding box`上的误差对各自的检测精确度影响是不一样的（小误差对应于大检测框，影响很小，而对于小检测框影响较大）。`YOLO`将物体大小的信息项（`w`和`h`）进行求平方根来改进这个问题。（注：这个方法并不能完全解决这个问题）。

综上，`YOLO`在训练过程中`Loss`计算如下式所示：

<center><img src="/images/YOLO/loss.jpg" width="60%"/></center>

## Limitations of YOLO
`YOLO`的局限性：
- `Bounding box`预测上的空间限制，因为每一个网格只预测`2`个`box`，并且最终只得到这个网格的目标类别，因为当目标的中心落入网格时，这个网格专门负责这一个目标。这种空间局限性限制了模型预测出那些挨得近的目标的数量（例如一个网格里可能会有多个小目标），**`YOLO`对小目标检测性能并不好**。
- **使用相对粗糙的特征去预测，影响了检测效果**。因为网络中对输入图像进行了多次下采样。
- **`loss`方程对小`bounding box`和大`bounding box`上误差的处理是相同的**。一般大边界框里的小误差是良性的，而小边界框里的小误差在`IoU`上有着更大的影响。**虽然采用求平方根方式，但没有根本解决问题**，从而降低了物体检测的定位准确性。

# Comparison to Other Detection Systems
`R-CNN`：生成`proposal`然后卷积网络提取特征，再做分类并且调整`bounding box`，这种复杂的流程中每一阶段都需要独立且精细地调整，因此速度慢。
`YOLO`：把空间限制放在网格的`proposal`上，这帮助缓解了同一个目标的重复检测。`YOLO`提取的`bounding box`更少（`98`，而选择性搜索`2000`）。
`Fast`和`Faster R-CNN`：致力于通过共享计算以及使用神经网络代替选择性搜索去提取`proposal`从而加速`R-CNN`，但是也只是在`R-CNN`基础上有一定的精度和速度的提升，仍然达不到实时。

**YOLO模型相对于之前的物体检测方法有多个优点：**
- **`YOLO`检测物体非常快**。因为没有复杂的检测流程，`YOLO`可以非常快的完成物体检测任务。标准版本的`YOLO`在`Titan X`的`GPU`上能达到`45 FPS`。更快的`Fast YOLO`检测速度可以达到`155 FPS`。而且，`YOLO`的`mAP`是之前其他实时物体检测系统的两倍以上。
- **`YOLO`可以很好的避免背景错误，避免产生`false positives`**。 不像其他物体检测系统使用了滑窗或`region proposal`，分类器只能得到图像的局部信息。`YOLO`在训练和测试时都能够看到一整张图像的信息，因此`YOLO`在检测物体时能很好的利用上下文信息，从而不容易在背景上预测出错误的物体信息。和`Fast R-CNN`相比，`YOLO`的将背景错误判断为目标的比例不到`Fast R-CNN`的一半。
- `YOLO`可以学到物体的泛化特征。当`YOLO`在自然图像上做训练，在艺术作品上做测试时，`YOLO`表现的性能比`DPM`、`R-CNN`等之前的物体检测系统要好很多。因为`YOLO`可以学习到高度泛化的特征，从而迁移到其他领域。

尽管YOLO有这些优点，它也有一些**缺点**：
- `YOLO`的物体检测精度低于其他`state-of-the-art`的物体检测系统。 
- `YOLO`容易产生物体的定位错误。 
- `YOLO`对小物体的检测效果不好（尤其是密集的小物体，因为一个栅格只能负责1个目标）。

# Reference
1. [论文原文](https://arxiv.org/abs/1506.02640)
2. [YOLO详解](https://zhuanlan.zhihu.com/p/25236464)
3. [YOLOv1论文理解](http://blog.csdn.net/hrsstudy/article/details/70305791)

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
