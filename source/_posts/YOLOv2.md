---
title: YOLO9000:Better, Faster, Stronger
date: 2018-05-02 15:04:13
categories: CNN
tags:
  - object detection
  - paper
---

> 这篇文章发表在2016年，提出了YOLO的第二个版本。YOLOv2在67FPS的检测速度下，可以在VOC2007上达到76.8的mAP。在40FPS的速度下，有78.6的mAP，比使用ResNet的Faster R-CNN和SSD更好。YOLO9000可以检测超过9000种目标。
<!-- more -->

# Introduction
当前目标检测数据集比起图像分类数据集而言，目标种类很少。因此本文利用已有的大量图像分类数据并扩展到目标检测系统中。将不同的数据集组合在一起，提出了一种联合训练算法，允许在检测和分类数据上训练目标检测器，利用检测图像学习精确地定位目标，同时利用分类图像扩展模型对多类别的识别能力。使用这种方法训练YOLO9000，可以获得一个可以检测9000种不同类别目标的实时检测器。

首先，在YOLO的基础上改进产生YOLOv2，然后使用组合的数据集和联合训练算法来训练ImageNet图像分类数据和COCO目标检测数据得到YOLO9000。

# Better
相比于最先进的目标检测系统而言，YOLO有一些缺点：与Fast R-CNN相比，YOLO产生大量的定位误差；与基于region proposal的方法相比，YOLO有较低的召回率。所以，本文的改进也主要是在这两方面。

YOLOv2的目标是保持很快的检测速度，达到更高的准确度。将过去工作中的思想与本文的新概念融合，以提高YOLO的性能。
- **Batch Normalization**：在YOLO的所有卷积层增加BN，mAP提升了2%。使用BN，可以去掉dropout而不过拟合。
- **高分辨率分类**：原始的YOLO训练的分类网络输入是224x224，并且为训练检测任务将分辨率增加到448。这意味着网络必须同时切换到学习目标检测并调整到新的输入分辨率上。
  - YOLOv2中，首先在ImageNet上，对分类网络在448x448的分辨率下进行10个epoch的微调。相当于给了网络一些时间去调整卷积核，使得在高分辨率输入下工作得更好。
  - 然后在检测数据集上微调网络。高分辨率的分类网络使得mAP有近4%的提升。
- **使用Anchor Box的卷积**：YOLO中使用全连接层直接在顶层的特征图上预测bounding box的坐标。但是Faster R-CNN使用RPN，在特征图的每一个位置预测anchor box的偏移量和置信度，预测偏移量简化了问题，并且使得网络更容易学习。本文移除掉YOLO中的全连接层，使用anchor box去预测bounding box。
  - 首先移除掉一个池化层，提高卷积层的输出分辨率，然后缩小网络将输入尺寸改为416而不是448×448，目的是希望得到的特征图中位置为奇数，这样就只有一个中心单元格。那些大目标倾向于占据图像的中心，所以在中心只有一个位置能很好预测这些目标。YOLO的卷积层下采样32倍，416的输入图像最终得到13x13的feature map。
  - 因为YOLO是由每个cell来负责预测类别，每个cell对应的2个bounding box 负责预测坐标 。YOLOv2中，不再让类别的预测与每个cell（空间位置）绑定一起，而是让全部放到anchor box中。
  <center><img src="/images/YOLOv2/box_predict.png" width="60%"/></center>
  - 跟随YOLO，在目标预测上，仍然预测ground truth和候选框的IoU，表示是否存在目标；类别预测上，预测类别概率，即存在目标的前提下目标的类别。
  - 使用anchor box准确度上有一点降低。YOLO对每幅图只预测98个box，但是使用anchor box后，每幅图预测超过1000个box。
- **维度聚类**：使用anchor box时遇到两个问题，一是box的尺寸是手工选取的，网络可以学习调整box，但假如选择更好的先验box，那么会使网络更容易预测出好的结果。
  - 取代手工选取anchor box尺寸，使用k-means在训练集的bounding box上进行聚类，自动地找到好的先验box。这里并不使用欧氏距离，因为这会导致较大的box比较小的box产生更大的误差。希望得到的先验box能提高IoU分数。因此定义距离度量为：$d(box,centroid) = 1 - IoU(box,centroid)$。
- **直接位置预测**：使用anchor box时遇到的第二个问题就是模型不稳定，特别是在早期的迭代时。不稳定性主要来源于box的(x,y)坐标。在RPN中，网络预测值$t_x$和$t_y$与中心坐标计算为$x=t_x*w_a-x_a$，$y=t_y*h_a-y_a$。$t_x= 1$的预测将使框向右移动anchor box的宽度， $t_x=-1$的预测将使其向左移动相同的量。 这种公式是不受约束的，因此不管预测box的位置，任何anchor box可以在图像中的任何点结束。使用随机初始化模型需要很长时间才能稳定到预测出可感知的偏移。
  - 本文遵循YOLO的方法并预测相对于网格单元位置的bounding box坐标。这将ground truth限制在0和1之间。我们使用逻辑激活函数来约束网络的预测落在该范围内。
  - 网络在feature map的每个网格单元上预测5个bounding box，每个box包含5个坐标：$t_x,t_y,t_w,t_h,t_o$。假如网格相对图像左上角偏移$(c_x,c_y)$，先验的bounding box宽高为$p_w,p_h$，则预测为：
    - $b_x = \sigma(t_x)+c_x$
    - $b_y = \sigma(t_y)+c_y$
    - $b_w = p_we^{t_w}$
    - $b_h = p_he^{t_h}$
    - $P_r(object)*IoU(b,object) = \sigma(t_o)$
      <center><img src="/images/YOLOv2/box.png" width="50%"/></center>
使用维度聚类和直接预测bounding box中心位置，比Faster R-CNN中的anchor box方法提升了近5%的mAP。
- **细粒度特征**（Fine-Grained Features）：改进的YOLO在13x13的feature map上预测，尽管这对大目标是足够的，但是对于定位更小的目标需要细粒度的特征。本文添加一个传递层（passthrough layer），与ResNet的恒等映射相似，传递层通过将相邻的特征图堆叠到不同的通道，而不是空间位置，将较高分辨率和较低分辨率的feature map相连。将26×26×512特征映射转换为13×13×2048特征映射，这样可以直接与原始的特征拼接。这一点改进获得了1%的性能提升。
- **多尺度训练**：由于使用anchor box，因此改变输入的分辨率为416。本文并不固定输入图像的尺寸，而是在每几次迭代中改变网络的输入图像尺寸。每10个batch，网络随机选择一个新的输入图像尺寸，由于下采样的因子是32，所以输入尺寸都是32的倍数：{320，352，...，608}。这个策略迫使网络学习在不同的输入分辨率下预测好的结果，意味着相同的网络可以实现不同分辨率的检测。在288的分辨率下，YOLOv2的速度是90FPS，mAP几乎和Faster R-CNN一样好，高分辨率下，在VOC2007数据集上YOLOv2达到78.6的mAP，仍然保证实时。

# Faster
由于VGG16还是过于复杂，YOLO框架中采用基于GoogLeNet的自定义网络，准确率略低于使用VGG16，但计算量更少。

本文设计了一个新的分类模型Darknet-19，共19个卷积层和5个最大池化层：
  <center><img src="/images/YOLOv2/darknet.png" width="50%"/></center>

- 与VGG模型相似，使用3x3的卷积核，并且在每个池化步骤后将通道数量加倍。
- 跟随Network in Network(NIN)，使用全局平均池化做预测，用1x1的卷积在3x3的卷积之间压缩特征。
- 使用批量归一化来稳定训练，加速收敛，正则化模型。

**训练分类任务**：在ImageNet的1000类数据集上训练160个epoch，输入的分辨率为224。然后在更大的尺寸448上，微调网络，训练10个epoch。

**训练检测任务**：修改分类网络，将最后的卷积层移除掉，添加3个3x3的卷积层，每一层有1024个卷积核，跟随1个1x1的卷积层，卷积核数量与检测任务需要的输出相对应。对于VOC数据集来说，预测包含5个bounding box，每一个包含5个坐标和20个类别，所以一共有125个卷积核(5x(20+5))。除此之外，还添加了一个传递层，从最后的3x3x512的层连接到倒数第二层，以便于模型可以使用细粒度特征。

# Stronger
训练期间，混合了检测和分类数据集。当网络看到标记为检测的图像时，可以反向传播整个YOLOv2的损失，当看到一个分类图像时，只传播网络结构中特定的分类部分损失。

这种方法带来了一些挑战，检测数据集只有常见目标和标签，而分类数据集有着更广泛的标签。ImageNet有着100多种狗，如果想要在两种任务的数据集上联合训练，需要一种连贯的方式去合并标签。

分类任务中一般使用softmax层，计算所有可能类别的概率，这假设了类别之间是互斥的。因此，需要一个多标签的模型来综合数据集，使类别之间不相互包含。

**层次分类**：ImageNet的标签是从WordNet中提取的，WordNet是一个语言数据库，用于构建概念及其关系。WordNet被构造为有向图，但是本文并不使用完整的图结构，而是根据ImageNet中的概念构建层次树来简化问题，最终的结果是WordTree。使用WordTree实现分类，预测在每个节点的条件概率，也就是预测给定同义词集合的每个下位词的概率。例如，在“terrier”节点，预测：$P_r(Norfolk~terrier|terrier)$，$P_r(Yorkshire~terrier|terrier)$...

如果想要计算一个特定节点的绝对概率，只需遵循通过树到达根节点的路径，并乘以条件概率。因此，如果我们想知道图片是否是诺福克梗犬，只需要计算：$$
P_r(Norfolk~terrier) = P_r(Norfolk~terrier|terrier) ... \times P_r(animal|physical~object)$$ 对于分类任务，假设图像中包含目标，因此$P_r(physical~object) = 1$。

Darknet-19模型的训练使用1000类ImageNet数据集构建的WordTree。为了构建WordTree1k，添加中间节点将标签空间从1000扩展到1369。在训练期间，沿着树传播ground truth标签，以便如果图像被标记为“诺福克梗犬”，它也被标记为“狗”和“哺乳动物”等。为了计算条件概率，模型预测了1369个值的向量，并且计算所有同义词集合的softmax，如下图：
  <center><img src="/images/YOLOv2/wordtree.png" width="60%"/></center>
以这种方式实现分类具有一些益处。在新的或未知的目标类别上性能降低微小。例如，如果网络看到一只狗的图片，但不确定它是什么类型的狗，它仍然会预测具有高置信度的“狗”，但在下义词（更为具体的内容，比如某种类型的狗）上具有较低的置信度。

这个公式也用于检测。不是假设每个图像都有一个目标，而是使用YOLOv2的目标预测器给出$P_r(physical~object)$的值。检测器预测bounding box和概率数。遍历树，在每个分裂中采用最高置信度路径，直到达到某个阈值，预测出目标类别。

**联合分类和检测**
要训练一个极大尺度的检测器，因此使用COCO数据集和来自完整ImageNet的前9000类创建组合数据集。还需要评估本文的方法，所以添加ImageNet还没有包括的类别。WordTree数据集相应的具有9418个类。ImageNet是一个更大的数据集，因此对COCO进行过采样来平衡数据集，使ImageNet与COCO数量只有4：1的倍数。使用这个数据集训练YOLO9000，使用基本的YOLOv2架构，但只有3个先验box，而不是5，以限制输出大小。

当网络看到检测图像时，反向传播正常的损失。对于分类损失，只在标签的相应层级或高于标签的相应层级反向传播误差。例如，如果标签是“狗”，会在树中“德国牧羊犬”和“金毛猎犬”的预测中分配误差，因为没有这些信息。

看到分类图像时，只反向传播分类损失，为了实现这一点，只找到对这个类别预测的概率最高的bounding box，并且只计算它的预测树上的损失。这里假设预测框与ground truth的IoU至少为0.3，并且基于该假设反向传播目标类别损失。
# Reference
1. [论文原文](https://arxiv.org/abs/1612.08242)
2. [YOLOv2](https://zhuanlan.zhihu.com/p/25167153)

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
