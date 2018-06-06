---
title: 两步实现博客评论的添加
date: 2018-06-01 13:47:21
categories: others
tags: hexo
---

> 只需要简单的两步，就能用Gitment为hexo搭建的博客实现评论功能。
<!-- more -->

# 注册 OAuth Application
注册一个[OAuth Application](https://github.com/settings/applications/new)。最重要的是填对`callback URL`，比如你的博客主页地址。其他的内容不重要。注册成功会得到一个client ID和一个 client secret。
# 引入 Gitment
- 创建一个repository，用来存储评论。这个可以新建，也可以是自己博客的那个仓库。我自己新建了一个仓库，名为`GitComment`。
- 引入 Gitment
  - 创建一篇博客，在本地的文件夹中会产生一个`a.md`的文件，也就是博文内容写入的文件。
  - 这个文件中前面都是自己的博文，只需要把下面的代码添加到这个文件的最后
```
<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: '<%= page.date %>', // 可选。默认为 location.href
  owner: '你的 GitHub ID',
  repo: '存储评论的 repo',
  oauth: {
    client_id: '你的 client ID',
    client_secret: '你的 client secret',
  },
})
gitment.render('container')
</script>
```
  - 在 themes\next\layout_third-party\comments 目录下修改gitments.swig，找到以下代码修改
```
function renderGitment(){
var gitment = new {{CommentsClass}}({
- id: window.location.pathname,
+ id: '{{ page.date }}',
owner: '{{ theme.gitment.github_user }}',
repo: '{{ theme.gitment.github_repo }}',
{% if theme.gitment.mint %}
lang: "{{ theme.gitment.language }}" || navigator.language || navigator.systemLanguage || navigator.userLanguage,
{% endif %}
```
  - 执行`hexo g -d`

要注意的是，每一篇需要添加评论的博文文章源码的最后，都要添加上面的代码。发布之后，还需点进博文网页，在评论处点击初始化。如果遇到有问题，可以参考[Gitment评论功能接入踩坑教程](https://www.jianshu.com/p/57afa4844aaa)。
  

# Reference
1. [使用 GitHub Issues 搭建评论系统](https://github.com/imsun/gitment#customize)
2. [Gitment评论功能接入踩坑教程](https://www.jianshu.com/p/57afa4844aaa)

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
