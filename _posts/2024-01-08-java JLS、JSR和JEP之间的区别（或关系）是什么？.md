---
title: java JLS、JSR和JEP之间的区别（或关系）是什么？
date: 2024-01-08 11:21:00 +0800

categories: [技术]
tags: [java]
---

> 参考：https://www.cnpython.com/java/1223952

英文：

- JLS-[*Java Language Specification*](https://docs.oracle.com/javase/specs/jls/se9/html/index.html)
- JSR-[*Java Specification Requests*](https://jcp.org/aboutJava/communityprocess/final/jsr376/index.html)
- JEP-[*JDK Enhancement Proposal*](http://openjdk.java.net/jeps/0)

在一般情况下：

- **规范**是*指定*（或定义）某物的文档
- 请求是要求某事的陈述（书面或口头）
- **提案是（书面或口头）提出要考虑的事项的声明**

## Java语言规范，JLS

这是Java语言的规范。JLS指定Java编程语言的语法和其他规则，说明什么是或不是有效的Java程序。它还规定了程序的含义；i、e.运行（有效）程序时会发生什么

## Java规范请求，JSR

JSR是作为Java社区过程（JCP）的一部分创建的文档，该过程为团队开发新规范设定了范围。这些规范（AFAIK）总是与Java相关的，但它们经常解决一些不会成为核心JavaSE或JavaEE技术的问题。典型的JSR主题材料是一种相对成熟的技术；i、 e.处于可以指定的状态的。（如果你试图过早地制定一个规范，那么你通常会得到一个糟糕的规范。其他因素也可能导致这种情况。）

## Java增强方案，JEP

JEP是一份提议增强Java核心技术的文档。这些建议通常针对尚未准备好具体说明的增强功能。正如JEP-0文件所述，JEP*可能要求探索新奇（甚至是“古怪”）的想法。一般来说，需要原型来区分可行和不可行的想法，并将它们澄清到可以生成规范的程度*

因此，JEP、JSR和规范之间的关系如下：

1. JEP提出并发展实验想法，使其*能够*具体化。并不是所有的都能开花结果。
2. JSR采用成熟的想法（例如，由JEP产生），并生成新的规范或对现有规范的修改。并不是所有的JSR都能实现。
3. 规范是JSR的常见工作产品。（其他包括接口的源代码和参考实现。）JLS是规范的*示例*。其他包括JVM规范、Servlet和JSP规范、EJB规范等等。
