---
title: Matplotlib简单使用
date: 2023-02-10 09:58:00 +0800
categories: [技术]
tags: [matplotlib]
---


-   菜鸟Matplotlib教程：[https://www.runoob.com/matplotlib/matplotlib-tutorial.html](https://www.runoob.com/matplotlib/matplotlib-tutorial.html)
-   Matplotlib 官网：[https://matplotlib.org/](https://matplotlib.org/)

## 绘制点和线

首先导入库

```python
import numpy as np
import matplotlib.pyplot as plt
```

通过两个坐标 **(0,0)** 到 **(6,100)** 来绘制一条线：

```python
x_points = np.array([0, 6])
y_points = np.array([0, 100])
plt.plot(x_points, y_points)  # 绘制
plt.show()  # 打开窗口，每次绘制好都要记得加上
```

只想绘制两个坐标点：

```python
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints, 'o')
```

这里使用了参数`'o'`，还有更多可使用的参数，详见文档。

绘制多条线：

```python
x1 = np.array([0, 6])
y1 = np.array([0, 100])
x2 = np.array([9, 4])
y2 = np.array([8, 100])
plt.plot(x1, y1, x2, y2)
```

绘制曲线：

```python
x = np.arange(0, 4 * np.pi, 0.1)  # 开始，结束，步长
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y, x, z)
```

## 更多常用方法

图像信息：

```python
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.plot(x, y, marker='o')  # 实心圆标志

plt.grid()  # 网格线
plt.title("TEST TITLE")  # 标题
plt.xlabel("x - label")  # 轴标签
plt.ylabel("y - label")
```

设置坐标轴的显示范围：

```python
plt.xlim(-math.pi, math.pi)
plt.ylim(-1, 1)
```

曲线信息：

```python
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
# 画出拟合的曲线，红色，线宽5
```

在图像上写点字：

```python
plt.text(0, 1, 'iter=%d, Loss=%.4f' % (t, loss.data.numpy()), fontdict={'size': 10, 'color': 'red'})
# 从坐标（0，1）的地方开始，写信息，字大小为10，颜色为红色
```

## 散点图

基础例子：

```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
plt.scatter(x, y)
```

加上点的大小：

```python
sizes = np.array([20, 50, 100, 200, 500, 1000, 60, 90])
plt.scatter(x, y, s=sizes)
```

加上颜色与颜色条：

```python
colors = np.array([0, 10, 20, 30, 40, 50, 60, 70])
plt.scatter(x, y, s=sizes, c=colors)
plt.colorbar()  # 颜色条
```

## 绘制多图

```python
# plot 1:
x1 = np.array([0, 6])
y1 = np.array([0, 100])
plt.subplot(1, 2, 1)  # 在（1x2）的分布中占第1个
plt.plot(x1, y1)
plt.title("plot 1")

# plot 2:
x2 = np.array([1, 2, 3, 4])
y2 = np.array([1, 4, 9, 16])
plt.subplot(1, 2, 2)  # 在（1x2）的分布中占第2个
plt.plot(x2, y2)
plt.title("plot 2")

plt.suptitle("subplot Test")  # 总标题
```

## imshow()函数

imshow()将数组的值以图片的形式展示出来，数组的值对应着不同的颜色深浅，而数值的横纵坐标就是数组的索引

**例子1**

```python
x = np.linspace(0, 10, 1000)  # 开始，结束，元素个数
I = x * x[:, np.newaxis]
plt.imshow(I)
cb = plt.colorbar()  # 添加颜色条，显示数值大小
```

-   其中，`x`是一维数组，`shape=(10000,)`，包含1000个元素，均匀分布从0到10
-   `x[:, np.newaxis]`是二维数组，`shape=(10000,1)`，对应元素与`x`相同
-   `I`是二维数组，包含1000\*1000个元素，相当于`x`和`x`矩阵乘

**例子2**

```python
x = np.arange(900)
y = x.reshape(30, -1)
plt.imshow(y)
cb = plt.colorbar()
```

-   其中，`x`是一维数组，包含900个元素，从0到900
-   `y`是二维数组，由`x`重塑，第一维为30，第二维自动计算

## 交互模式

显示模式默认为**阻塞（block）模式**：

-   `plt.plot(x)`或`plt.imshow(x)`之后需要plt.show()后才能显示图像
-   在`plt.show()`之后，程序会暂停而不是继续执行下去，必须关掉才能打开下一个新的窗口

使用`plt.ion()`这个函数，使显示模式转换为**交互（interactive）模式**：

-   `plt.plot(x)`或`plt.imshow(x)`直接显示图像
-   图像会一闪而过，并不会常留，即使在脚本中遇到`plt.show()`，代码还是会继续执行

那么，在交互模式中如何解决图像一闪而过的问题？两种方式：

1.   让程序暂停一会

     ```python
     plt.pause(10)  # 暂停10秒
     ```

2.   在所有图像画好之后返回阻塞模式

     ```
     plt.ioff()
     plt.show()
     ```

例子：

```python
x = np.linspace(0, 10, 1000)
i1 = x * x[:, np.newaxis]
i2 = np.sin(x) * x[:, np.newaxis]

plt.ion()  # 打开交互模式
plt.figure()  # 图片一窗口
plt.imshow(i1)
plt.figure()  # 图片二窗口
plt.imshow(i2)

# 显示前关掉交互模式
plt.ioff()
plt.show()
```


