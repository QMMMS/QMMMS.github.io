---
title: 加速Stack Overflow方法
date: 2023-02-03 10:42:00 +0800
categories: [技术]
tags: [stack overflow]
---

[Stack Overflow](https://stackoverflow.com/)是国外一个与程序相关的IT技术问答网站，类似于国内的[segmentfault](https://segmentfault.com/)。然而打开Stack Overflow速度非常慢，如何解决？

## 原理

很多网站，尤其是国外网站，包括Stack Overflow，为了加快网站的速度，都使用了 Google 的 CDN。 但是在国内，由于某些原因，导致全球最快的 CDN 变成了全球最慢的。

>   *CDN* (内容分发网络) 指的是一组分布在各个地区的服务器。这些服务器存储着数据的副本，因此服务器可以根据哪些服务器与用户距离最近，来满足数据的请求。

将 Google 的 CDN 替换成国内的，就可以解决。

## 安装浏览器插件

>   Replace Google CDN 的 [GitHub链接](https://github.com/justjavac/ReplaceGoogleCDN)

1.  下载 [ReplaceGoogleCDN](https://github.com/justjavac/ReplaceGoogleCDN/archive/master.zip) 然后解压，找到 `extension` 子目录
2.  打开 Chrome，输入: `chrome://extensions/`
3.  勾选 Developer Mode
4.  选择 Load unpacked extension... 然后定位到刚才解压的文件夹里面的 extension 目录，确定
5.  这就安装好了，去掉 Developer Mode 勾选。


>   扩展下载方式（原理：克隆国内镜像源代码）
>
>   ```shell
>   # 克隆源代码
>   git clone -b master https://gitee.com/mirrors/replacegooglecdn.git --depth=1 --progress
>   # 或者
>   git clone -b master https://gitcode.net/mirrors/justjavac/replacegooglecdn.git --depth=1 --progress
>
>   # 更新源代码
>
>   git -C replacegooglecdn pull  --depth=1 --progress  --rebase=true
>
>   ```
