---
title: GCC、CMake、CMakelist、Make、Makefile、Ninja啥关系？一图讲透！
date: 2023-08-26 11:21:00 +0800

author: qrsdz

categories: [经验与总结]
tags: [转载, 经验]
img_path: "/assets/img/posts/2023-08-26-GCC、CMake、CMakelist、Make、Makefile、Ninja啥关系？一图讲透！"
---

> 转自微信，[原文链接](https://mp.weixin.qq.com/s/LqBjfLGtYUxW0ljwG_w4NA)
{: .prompt-tip }

## 一、GCC的诞生！

1985年，一个伸手不见五指的黑夜，一位满脸胡须、五短身材、头发长乱的怪人开始编写一套C语言编译器。这个怪人后来被称作“最伟大的程序员”、“自由软件之父”，他就是 Richard Stallman。经过2年时间没日没夜的苦练，胡须更长、头发更乱的Richard Stallman完成了C语言编译器初版，称之为GNU C Complier，也就是现在大名鼎鼎GCC的来源。后来GCC快速发展，现在GCC除了能编译C语言，还能编译C++、Fortran、Pascal、Objective-C、Java以及Ada 等语言。上世纪80、90年代，个人计算机CPU频率、硬盘大小、内存大小都是以MB为单位，甚至KB为单位，需要编译的程序文件个数较少，用GCC命令也足以应对。

## 二、新的麻烦出现了! (make & makefile)

随着计算机的性能提升和普及，一个软件工程包含的源文件越来越多，比如linux，采用gcc命令逐个手动去编译，很容易混乱而且工作量大，会让人抓狂。

于是乎，Richard Stallman和 Roland McGrath 共同开发了make 和makefile来解决这个问题。make工具可以看成是一个智能的批处理工具，它本身并没有编译和链接的功能，而是用类似于批处理的方式—通过调用makefile文件中用户指定的命令来进行编译和链接的。makefile是什么？makefile就是一个脚本文件，简单的说就像一首歌的乐谱，make工具就像指挥家，指挥家根据乐谱指挥整个乐团怎么样演奏，make工具就根据makefile中的命令进行编译和链接的。makefile命令中就包含了调用gcc（也可以是别的编译器）去编译某个源文件的命令。

## 三、新的麻烦又出现了！（Cmake & CMakelist）

makefile在一些简单的工程下，完全可以人工手写，但是当工程非常大的时候，手写makefile也是非常麻烦。而且陆陆续续出现了各种不同平台的makefile，有GNU make、QT 的 qmake，微软的 MSnmake，BSD Make（pmake），Makepp等等。这些 Make 工具遵循着不同的规范和标准，所执行的 Makefile 格式也千差万别。这样就带来了一个严峻的问题：如果软件想跨平台，必须要保证能够在不同平台编译。而如果使用上面的 Make 工具，就得为每一种标准写一次 Makefile ，这将是一件让人抓狂的工作，如果换了个平台makefile又要重新修改。

于是乎，这时候就出现了Cmake这个工具，Cmake就可以更加简单地生成makefile文件给上面那个make用。当然cmake还有其他功能，就是可以跨平台生成对应平台能用的makefile，你不用再自己去修改了。cmake正是makefile的上层工具，它们的目的正是为了产生可移植的makefile，并简化自己动手写makefile时的巨大工作量。可是cmake根据什么生成makefile呢？它又要根据一个叫CMakeLists.txt文件（学名：组态档）去生成makefile。到最后CMakeLists.txt文件谁写啊？当然是你自己手写的。

## 四、新的麻烦又又又出现了！(Ninja)

随着软件工程越来越大，越来越复杂，make的执行速度慢的问题越来越严重了。

于是乎，Google的一名程序员推出的注重速度的构建工具Ninja。Ninja 舍弃了各种高级功能，语法和用法非常简单，给它指定好了具体详细要做什么，所以启动编译的速度非常快。根据实际测试：在超过30,000个源文件的情况下，也能够在1秒钟内开始进行真正的构建。与之相比，通过资深工程师进行编写的Makefiles文件也需要10-20秒才能开始构建。

## 五、目前完美的结局！

Cmake可以生成 .ninja 和.makefile 的工具。因为担心很多人不熟悉makefile文件和ninja文件的写法，所以cmake只需要用户通过对源码文件的简单描述（就是CMakeLists.txt文件），就能自动生成一个project的makefile文件或者ninja文件，然后就可以通过ninja或者make进行启动编译了。

![](make.png)
