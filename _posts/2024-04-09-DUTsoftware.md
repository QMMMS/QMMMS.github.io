---
title: DUT软院与建艺学院合作项目回顾：软著申请与Qt技术
date: 2024-04-09 12:21:00 +0800

img_path: "/assets/img/posts/2024-04-09-DUTsoftware"
categories: [经验与总结]
tags: [dut, 经验]
---

## 成果与总体情况

一共做了四个项目，并申请了四项软件著作权，项目代码闭源，软著名称如下：

- 视听交互作用下的声景观审美偏好实验系统。
- 道路交通噪声感知下城市街道步行空间多环境场景的声舒适性评估系统。
- 多声源多场景综合感知下临道路商业步行街声景的舒适性评估系统。
- 基于多感官健康舒适性能的高校图书馆使用后评估及建筑环境综合分析系统。

软件快照：

![](rz1.png)

![](rz2.png)

![](rz3.png)

![](rz4.png)

![](rz5.png)

名义上说是**软件学院**与**建筑与艺术学院**合作项目（其中几个项目可能涉及北京某公司）。项目的idea来源可能是建艺学院A老师的研究生产出研究成果，A老师也需要成果，不过需要完成的软件较为简单，不必委托软件公司来完成，委托计算机专业或者软件工程专业的同学来做可能更快（也更便宜）。于是A老师找到了软件学院的B老师，B老师想到了我并给我看了**需求说明书**。由于项目需求较为简单，不需要联网，不必使用数据库，欣然接收了这个项目。

项目成员构成方面，涉及的人员不算多。好听一点说，如果要分析stake holder，A老师是甲方，虽然用到了研究生的模型，研究生并不直接与我沟通，在描述需求与提出修改意见时，都是A老师直接与我沟通。我担任的角色包括：

- **产品经理**：负责产品的策划、设计、开发、实施和维护。
- **业务分析师**：对于一个像这样的小项目来说，甲方给的**需求说明书**往往只是需求的开始，后续的详细需求都以**微信语音**的方式给出，得从几十段语音中分析出甲方真正想要的功能，还得往要编写的软件上靠。
- **UI/UX 设计师**：负责用户界面（UI）和用户体验（UX）的设计。Qt不像网页那样有现成的UI模板，需要自己设计外观怎样好看而且专业。一般来说当我们软件学生自己鼓捣技术的时候，只注重功能实现，没有关注软件外观，但是在现实的软件项目中，**好看、简洁和专业的UI设计是重要的**，这是我写软件时遇到的最大挑战之一（毕竟没学过设计），好在可以借鉴一些前端框架的页面设计。
- **前端开发工程师**：负责开发用户界面和前端逻辑。Qt的设计器可以将组件拖拽到页面上面，一些简单的页面设计可以通过组件拖拽来完成，另外一些与用户交互相关的复杂的组件需要通过代码来生成。
- **后端开发工程师**：负责处理服务器端逻辑和数据处理，开发后端功能和数据库管理，使用各种编程语言和框架。严格来说这个软件**没有后端**，因为它不联网，所有的功能只在用户机器里面的前端软件中完成，数据库可以有，只不过调用的是存放在用户机器里面的**SQLite**。此外，研究生给出的模型是较为简单数学模型，不是需要做大量运算的深度学习模型，所以说数据处理逻辑不需要交给服务器。
- **数据库管理员（DBA）**：负责数据库设计。严格来说，没有使用数据库的必要。但是为了在文档中能够多写一点软件技术，在某几个项目中使用了**SQLite**。
- **测试工程师**：负责开发测试策略、执行测试。包括每个功能写好执行的**单元测试**，多个功能写好后放在一起执行**集成测试**，以及集成后验证需求是否实现的**系统测试**。由于一些模块是相互关联的，在大改完某一个模块后，**回归测试**是必要的。由于软件学院和建筑与艺术学院不在同一个校区，我和A老师执行**alpha测试**不现实，就直接给老师**beta测试**了。
- **简单点说，开发团队就我一个人。**

关于使用的开发方式，说的难听点是小作坊式开发，说的好听点是**敏捷开发**：

- **个体和互动**高于流程和工具：没有流程，微信聊天获取需求和修改意见。
- **工作的软件**高于详尽的文档：没有开发文档，只有软件。（不过申请软著时是需要写文档的）
- **客户合作**高于合同谈判：口头承诺，微信密切合作。
- **响应变化**高于遵循计划：甲方需求天天变。

此外，由于是第一次给建艺学院老师写软件，采用了**快速原型模型**的思想，每个阶段都给甲方一个软件，使得需求可以逐步稳定：

1. 第一版是基本没有UI设计的纯功能Demo，确认核心功能模块是否符合需求。
2. 第二版是设计了UI的做好的软件。
3. 第三、四、五、六。。。。N版软件是在整体框架上的小改动，基本上是每做一版甲方就有新需求，然后不断迭代。。。。直到改好。

关于项目完成中与完成后的反思：

- **重构**。由于几个软件整体的风格是相似的，一些组件可以重用。但是遇到了问题：类名需要修改，但是**Qt的重构功能做得不好**，由于在Qt中类名不仅会在代码中出现，也会在UI文件中出现，会导致代码文件中改了类名，UI文件中还要自己手动改的情况。另外，**Qt也不支持一键改变项目名称**，还要自己手动设置。这就让开发者重新权衡修改类名的得失：改类名浪费时间而且项目可能会崩，不改则很恶心人，功能和类名对不上，不过由于开发就我一个人，恶心恶心也就完了，看着逐渐成长的“史山”，希望做完项目后再也不要碰他。
- **自动化测试工具**。由于功能较少，没有使用自动化测试工具，完全自己手动测试，就是点击特定的按钮组合，查看结果是否符合预期。不过到项目后期，出现了多次的模块微调而需要**回归测试**，明明是一个很小的微调但是仍然需要测试来保证软件运行正确，基于用户界面（GUI）的自动化测试工具是需要的。大三下正好在学软件测试，**Unified Functional Testing**（简称UFT）可以是一个不错的工具，可以使用录制功能，建立测试用例脚本。
- **不被使用的功能**。对于一个任务，在两个使用场景中有两个功能，每个使用场景使用对应的功能是最方便的。但是到头来，这是我的想法，在软件交付几版后，才发现用户只是“别扭地”只使用一种功能，另一种功能被完全抛弃了（但是我写了好久）
- **创新**。对于一个高校老师和研究结果而言，创新是十分重要的，然而老师希望我编写的软件同样“创新”——每一个项目都使用不同的技术方案编写。这对扩展技术栈确实很好，但是我的成本太大了——啥叫“使用不同的技术方案编写”？放弃Qt使用另一个编程框架？咋不叫我换种语言呢？工程和研究确实不一样，我们程序员总想总结出几种设计模式来复用，总想找到“最佳实践”来复用。最后和老师以一种简单的方式解释清楚了。
- **熟练度**。非功能性的细节改动可能远远超出做功能花费的时间。项目后期往往处于一种换字体、换颜色、调整组件间距的小改动中。夸张点讲，需要对实现的界面实现**像素级的把握**，如果换用一个没那么熟练的框架，那我肯定是做不到这种细微调整的。

## 软著申请

申请软著需要的文档：

- **使用说明书**：文字和界面配合说明，要求说清楚软件保护的内容和使用方式即可，无过多要求；
- **源程序**：最多提交60页，每页不少于50行（换句话说，不少于3000行，这四个项目中少的3500多行，多的10000多行）；若多于60页，提交连续的前30页和连续的后30页；若不够60页，全部提交。不留空行。
- **软件信息登记表**。

具体申请细节上，一般的好大学都会有学校专人负责软著申请流程的，只需要把资料交给他们就行了。关于署名，由于公司和学校要求，应该说软院内大部分软著的署名都是大工学校本身，不过可以出一个证明书证明这个软件是谁负责编写完成的。

## Qt技术细节

### 常用操作

快捷键F4实现头文件和源代码文件的快速跳转。

### 快速搭建页面和给组件添加方法

善用横向和纵向弹簧，可以实现窗口大小自适应的组件和页面。

例如按钮，右键，转到槽函数，可以给组件添加方法。

### 添加资源文件

新建文件，Qt，Qt resource file，然后添加资源文件，可以添加图片、音频等。

### 打包成可执行的exe

> 参考：https://blog.csdn.net/zlpng/article/details/130773437

1. 从Debug模式切换到Release模式。
2. 编译或者运行一次。
3. 从编译目录下把对应的exe文件拷贝到自己指定的目录中。
4. 通过导入相应的依赖，命令：`windeployqt <exe 路径>`

### QLabel设置

如果发现QT 切换界面时（例如在win下 全屏显示 showFullScreen），label 如果开启自动换行功能会出现布局错乱，解决方法是关闭自动换行 QLabel->setWordWrap(false);

我们在使用QLabel进行内容显示的时候，通常有一个最大长度，超过这个长度怎么办呢？一个QLabel不可能显示无限长的字符串啊，这时候我们可以考虑，如果字符串长度太长的时候，我们就显示其中的一部分，剩下的用…显示（这个功能只能自己手动写，Qt没有写好给你用）。

> 比如说我们要显示“1234567890abcdefghijklmnopqrstuvwxyz”,能不能显示成“12345…”或者“123…xyz”这样的呢？参考：https://blog.csdn.net/qq_43627385/article/details/107354487

```cpp
QString newStrMsg = "1234567890abcdefghijklmnopqrstuvwxyz";

QFontMetrics fontWidth(ui->noteValueLabel->font());//得到每个字符的宽度
QString elideNote = fontWidth.elidedText(newStrMsg, Qt::ElideRight, 150);//最大宽度150像素

ui->noteValueLabel->setText(elideNote);//显示省略好的字符串
ui->noteValueLabel->setToolTip(newStrMsg);//设置tooltips
```

### 设置应用程序图标

> 参考：https://blog.csdn.net/qq_40170041/article/details/129703899

1. 找一个喜欢的图像，转成ico格式，放在工作目录下。
2. 在pro文件中添加下面一句 `RC_ICONS = myico.ico`
3. 重新编译运行，就能看到图标。

### 项目名称修改

> 参考：https://download.csdn.net/blog/column/9227302/127242719

1. 更改项目代码所在的文件夹名（不改也可以）
2. 打开代码文件夹，删掉.user文件，更改.pro文件的文件名。
3. 双击打开.pro文件，重新配置项目（只要Qt版本兼容，应该都能成功配置）。

### 视频播放

> 参考：https://blog.csdn.net/bx1091182836/article/details/128275073

视频播放的核心逻辑是，让一个虚拟的`QMediaPlayer`拥有具体的`QVideoWidget`作为视频输出组件和具体的`QAudioOutput`作为音频输出组件，当视频播放时，接收当前帧改变的信号，并自定义一个槽函数让（充当视频进度条的）水平滑块得知。同样，当用户拖动水平滑块的时候，也让`QMediaPlayer`得知，改变视频进度。

因为要引入视频播放器的模块，我们在pro中引入对应的模块

```
 QT += multimedia multimediawidgets
```

可能会用到的头文件：

```cpp
#include <QWidget>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QAudioOutput>
#include <QPushButton>
#include <QSlider>
#include <QHBoxLayout>
#include <QLabel>
#include <QTime>
#include <QFileDialog>
#include <QMovie>
```

初始化视频播放组件：

```cpp
ui->vidWidget->setFixedSize(1000,650);	// 在设计器中拖放好的普通 widget
audiooutput = new QAudioOutput();
player = new QMediaPlayer;
player->setPlaybackRate(1.0);//默认1倍速播放

QVBoxLayout* vlayout = new QVBoxLayout(ui->vidWidget);

videoWidget = new QVideoWidget(this);
vlayout->addWidget(videoWidget, 10);
player->setVideoOutput(videoWidget);//设置播放窗口
player->setAudioOutput(audiooutput);//设置声音
audiooutput->setVolume(1);//初始音量为1

player->setSource(QUrl::fromLocalFile("本地视频路径"));
player->pause();//初始暂停

// 当 player 的duration改变，即切换了其他视频，需要让本类的其他组件得知
// 当 player 的position改变，即视频播放时当前帧改变，也需要让本类的其他组件得知
connect(player, &QMediaPlayer::durationChanged, this, &OutputWidget::getduration);
connect(player, &QMediaPlayer::positionChanged, this, &OutputWidget::VideoPosChange);

ui->horizontalSlider_2->setRange(0,100);//音量水平滑块，范围是0~100
ui->horizontalSlider_2->setValue(100);//当前音量
```

当 player 的duration改变，即切换了其他视频，需要让本类的其他组件得知：

```cpp
void OutputWidget::getduration(qint64 duration)
{
    qint64 vid_duration = duration;  // 注意数据类型，因为以帧为单位因此这个数可能很大
    ui->horizontalSlider->setRange(0,duration);  // 视频进度条范围
    QTime currentTime(0, 0, 0, 0);
    currentTime = currentTime.addMSecs(duration);
    totalTime = currentTime.toString("mm:ss");
    ui->label_3->setText("00:00 / "+totalTime);  // 视频进度条旁边的标签
}
```

当 player 的position改变，即视频播放时当前帧改变，也需要让本类的其他组件得知：

```cpp
void OutputWidget::VideoPosChange(qint64 pos)
{
    QTime currentTime(0, 0, 0, 0); // 初始化一个时间为0的QTime对象
    currentTime = currentTime.addMSecs(pos); // 将当前播放时间（毫秒）添加到 QTime 对象中
    QString currentFormattedTime = currentTime.toString("mm:ss"); // 转换 QTime 对象到分:秒格式字符串
    ui->label_3->setText(currentFormattedTime + " / " + totalTime);

    if(ui->horizontalSlider->isSliderDown()){
        return;  // 如果用户现在正在拖动视频进度条，则不要动进度条
    }
    ui->horizontalSlider->setSliderPosition(pos);
}
```

在用户拖动视频进度条后，改变视频进度。

```cpp
void OutputWidget::on_horizontalSlider_valueChanged(int value)
{
    if(ui->horizontalSlider->isSliderDown()){
        player->setPosition(value);
    }
}
```

在用户拖动声量条后，改变声量大小。

```cpp
void OutputWidget::on_horizontalSlider_2_valueChanged(int value)
{
    if(ui->horizontalSlider_2->isSliderDown()){
        volume = (double)value / 100;
        audiooutput->setVolume(volume);
    }
}
```

### 操作SQLite数据库

> 参考：https://zhuanlan.zhihu.com/p/615519914

Sqlite 数据库作为 Qt 项目开发中经常使用的一个轻量级的数据库，可以说是兼容性相对比较好的数据库之一（Sqlite就像Qt的亲儿子，如同微软兼容Access数据库一样）。Qt5 以上版本可以直接使用（Qt自带驱动），是一个轻量级的数据库，概况起来具有以下优点：

- SQLite 的设计目的是嵌入式 SQL 数据库引擎，它基于纯C语言代码，已经应用于非常广泛的领域内。
- SQLite 在需要长时间存储时可以直接读取硬盘上的数据文件（.db），在无须长时间存储时也可以将整个数据库置于内存中，两者均不需要额外的服务器端进程，即 SQLite 是无须独立运行的数据库引擎。
- 源代码开源，你可以用于任何用途，包括出售它。
- 零配置 – 无需安装和管理配置。
- 不需要配置，不需要安装，也不需要管理员。
- 同一个数据文件可以在不同机器上使用，可以在不同字节序的机器间自由共享。
- 支持多种开发语言，C, C++, PHP, Perl, Java, C#,Python, Ruby等。

**引入SQL模块**：在Qt项目文件(.pro文件)中，加入SQL模块：

```text
QT += sql
```

**引用头文件**：在需要使用SQL的类定义中，引用相关头文件。例如：

```text
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
```

**建立数据库**：

```cpp
QSqlDatabase database;

if (QSqlDatabase::contains("qt_sql_default_connection"))
{
    database = QSqlDatabase::database("qt_sql_default_connection");
}
else
{
    // 建立和SQlite数据库的连接
    database = QSqlDatabase::addDatabase("QSQLITE");
    // 设置数据库文件的名字
    database.setDatabaseName("MyDataBase.db");
}
```

**打开数据库**：使用 open() 打开数据库，并判断是否成功。注意，在第一步检查连接是否存在时，如果连接存在，则在返回这个连接的时候，会默认将数据库打开。

```cpp
if (!database.open())
{
    qDebug() << "Error: Failed to connect database." << database.lastError();
}
else
{
    // do something
}
```

**关闭数据库**：数据库操作完成后，最好关闭。

```cpp
database.close();
```

**创建表**：创建一个名为student的表，表格包含三列，第一列是id，第二列是名字，第三列是年龄。

```cpp
// 用于执行sql语句的对象
QSqlQuery sqlQuery;
// 构建创建数据库的sql语句字符串
QString createSql = QString("CREATE TABLE student (\
                          id INT PRIMARY KEY NOT NULL,\
                          name TEXT NOT NULL,\
                          age INT NOT NULL)");
sqlQuery.prepare(createSql);
// 执行sql语句
if(!sqlQuery.exec())
{
    qDebug() << "Error: Fail to create table. " << sqlQuery.lastError();
}
else
{     qDebug() << "Table created!";
}
```

**插入单行数据**：在刚才创建的表格中，插入单行数据。

```cpp
// 方法一：使用 bindValue 函数插入单行数据
QSqlQuery sqlQuery;
sqlQuery.prepare("INSERT INTO student VALUES(:id,:name,:age)");
sqlQuery.bindValue(":id", max_id + 1);
sqlQuery.bindValue(":name", "Wang");
sqlQuery.bindValue(":age", 25);
if(!sqlQuery.exec())
{
    qDebug() << "Error: Fail to insert data. " << sqlQuery.lastError();
}
else
{
	// do something    
}

// 方法二：使用 addBindValue 函数插入单行数据
QSqlQuery sqlQuery;
sqlQuery.prepare("INSERT INTO student VALUES(?, ?, ?)");
sqlQuery.addBindValue(max_id + 1);
sqlQuery.addBindValue("Wang");
sqlQuery.addBindValue(25);
if(!sqlQuery.exec())
{
    qDebug() << "Error: Fail to insert data. " << sqlQuery.lastError();
}
else
{
	// do something    
}

// 方法三：直接写出完整语句
if(!sql_query.exec("INSERT INTO student VALUES(3, \"Li\", 23)"))
{
    qDebug() << "Error: Fail to insert data. " << sqlQuery.lastError();
}
else
{
	// do something 
}
```

**查询全部数据**

```cpp
QSqlQuery sqlQuery;
sqlQuery.exec("SELECT * FROM student");
if(!sqlQuery.exec())
{
	qDebug() << "Error: Fail to query table. " << sqlQuery.lastError();
}
else
{
	while(sqlQuery.next())
	{
		int id = sqlQuery.value(0).toInt();
		QString name = sqlQuery.value(1).toString();
		int age = sqlQuery.value(2).toInt();
		qDebug()<<QString("id:%1    name:%2    age:%3").arg(id).arg(name).arg(age);
	}
}
```

**更新数据（修改数据）**

```cpp
QSqlQuery sqlQuery;
sqlQuery.prepare("UPDATE student SET name=?,age=? WHERE id=?");
sqlQuery.addBindValue(name);
sqlQuery.addBindValue(age);
sqlQuery.addBindValue(id);
if(!sqlQuery.exec())
{
    qDebug() << sqlQuery.lastError();
}
else
{
    qDebug() << "updated data success!";
}
```

### 常用数据结构操作方法

> 参考：
>
> - https://blog.csdn.net/ken2232/article/details/131715561
> - https://blog.csdn.net/linvisf/article/details/124706413

```cpp
QString str = "User: ";
str.append(userName);  // 拼接
str.append("\n");

QString x = "Nine pineapples";
QString y = x.mid(5, 4);  // y == "pine" 
QString z = x.mid(5);  // z == "pineapples"

QString x = "Pineapple";  
QString y = x.left(4);      // y == "Pine"

QString x = "sticky question";  
QString y = "sti";  
x.indexOf(y);               // returns 0  
x.indexOf(y, 1);            // returns 10  
x.indexOf(y, 10);           // returns 10  
x.indexOf(y, 11);           // returns -1

double value = 35468 * 1.0 / 1000;
qDebug() << "保留两位小数：" << QString::number(value, 'f', 2);
```

### 设置页面全屏模式

> 参考：https://blog.csdn.net/futurescorpion/article/details/132065558

```cpp
mainWindow->setWindowState(Qt::WindowFullScreen);
```

### 样式表语法

> 参考：
>
> - https://blog.csdn.net/m0_73443478/article/details/129046751
> - https://blog.csdn.net/m0_60259116/article/details/127812977
> - https://blog.csdn.net/qq_39736982/article/details/132102527

Qt样式表是一个可以自定义部件外观的强大机制，样式表的概念、术语、语法均受到HTML层叠样式表(CSS)的启发。

与HTML的CSS类似，Qt的样式表是纯文本的格式定义，在应用程序运行时可以载入和解析这些样式定义。使用样式表可以定义各种界面组件(QWidget类及其子类)的样式，从而使应用程序的界面呈现不同的效果。很多软件具有换肤功能，使用Qt的样式表就可以很容易的实现这样的功能。

多多翻看Qt官方文档，所有控件都有案例：在索引栏输入Qt style。

**样式规则**：每个样式规则由选择器和声明组成。

例如：`QPushButton{color:red;background-color:white}`

- QPushButton是选择器
- {color:red}是声明
- color是属性
- red是值
- 声明中的多组"属性 ： 值"列表以分号；隔开

**选择器类型**

| 选择器             | 示例                      | 描述                                   |
| ------------------ | ------------------------- | -------------------------------------- |
| 通用选择器         | *                         | 匹配所有控件                           |
| 类型选择器         | QPushButton               | 匹配给定类型控件，包括子类             |
| 类选择器           | .QPushButton              | 匹配给定类型控件，不包括子类           |
| 属性选择器         | QPushButton[flat="false"] | 匹配给定类型控件中符合[属性]的控件     |
| ID选择器           | QPushButton#closeBtn      | 匹配给定类型，且对象名为closeBtn的控件 |
| 子对象选择器       | QDialog>QPushButton       | 匹配给定类型的直接子控件               |
| 子孙对象选择器     | QDialog QPushButton       | 匹配给定类型的子孙控件                 |
| 辅助(子控件)选择器 | QComboBox::drop-down      | 复杂对象的子控件                       |
| 伪状态选择器       | QPushButton:hover         | 控件的特定状态下的样式                 |

选择器可以解决几个样式规则对相同的属性指定不同的值时会产生冲突。如：

```css
QPushButton#okButton{color:gray}
QPushButton{color:red}
```

冲突原则：特殊的选择器优先。此例中QPushButton#okButton代表的是单一对象，而不是一个类的所有实例，所以okButton的文本颜色会是灰色的。同样的有伪状态的比没有伪状态的优先。

**伪状态**：选择器可以包含伪状态来限制规则在部件的指定状态上的应用。伪状态在选择器之后，用冒号隔离。如鼠标悬停在按钮上时被应用：

```css
QPushButton:hover{color:white}
```

**盒子模型**：使用样式表时，每个部件被看作拥有4个同心矩形的盒子，四个矩形的内容分别为内容(content)、填衬(padding)、边框(border)、边距(margin）。边距、边框宽度和填衬等属性的默认值都是0，这样四个矩形正好重叠。

![](qbox.png)

**案例：单独设置某条边框的样式**

```css
QLabel
{
    border-left: 2px solid red;
    border-top: 2px solid black;
    border-right: 2px solid blue;
    border-bottom-color: transparent;   /*下边框透明，不显示*/
}
```

**案例：设置边框半径(圆角)**

```css
QLabel
{
    border-left: 2px solid red;
    border-top: 2px solid black;
    border-right: 2px solid blue;
    border-bottom: 2px solid yellow;

    border-top-left-radius: 20px;  /*设置左上角圆角半径，单位 px 像素*/
    border-top-right-radius: 15px;
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 5px;    
    /*border-radius: 20px;*/
}
```

**案例：设置QWidget边框样式**

```css
QWidget
{
    font-family:Microsoft YaHei UI;
    background:#ffffff;
    /*border:3px solid rgba(207, 209, 208, 170);设置整体边框*/
    border-bottom:  3px solid rgba(207, 209, 208, 170);/*设置底部边框*/
	border-top:  3px solid rgba(207, 209, 208, 170)/*设置顶部边框*/
}
```

### 设置背景图片

> 参考：https://blog.csdn.net/weixin_45866980/article/details/133314328

方案一：在背景widget的**styleSheet**中编辑样式表，添加资源，选择blackground-image。但是这么做会使这个应用页面的所有widget都包含该背景图。

方案二：拖一个“Frame”控件进入子界面，然后把按钮之类的控件都拖进去（可以把Frame的大小调整到和界面一样大的大小，对齐后边界设置为0），然后调出来styleSheet，编辑样式表：

```css
#frame{border-image: url(:/prefix/test.png)}
```

### 文件夹操作

> 参考：https://blog.csdn.net/m0_63647324/article/details/132499329

**创建文件夹**

```cpp
QString str =  "/test";    
QString path = QCoreApplication::applicationDirPath();
QString test = path+str;   
QDir *folder = new QDir;
//判断创建文件夹是否存在
bool exist = folder->exists(test);
if(exist)
{
    QMessageBox::warning(this,tr("创建文件夹"),tr("文件夹已经存在！"));
}
else //如果不存在，创建文件夹
{
    //创建文件夹
    bool ok = folder->mkpath(test);
    //判断是否成功
    if(ok)
    {
        QMessageBox::warning(this,tr("创建文件夹"),tr("文件夹创建成功！"));
    }
    else
    {
        QMessageBox::warning(this,tr("创建文件夹"),tr("文件夹创建失败！"));
    }
}
```

**用户自己选择文件夹路径**

```cpp
QString dir_path=QFileDialog::getExistingDirectory(nullptr,"choose directory","C:/");
```

- 第一个参数是指定的父窗口
- 第二个参数是窗口的标题
- 第三个参数是初始从哪里开始查找
- 返回值为你选择的文件夹路径

**获取文件夹下的子文件夹**

```cpp
QDir directory("D:/FastCharge"); // 替换为你的目录路径
QStringList subDirectories;

// 遍历所有子目录
for(const QFileInfo& info : directory.entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot)){
    if(info.isDir()){
        subDirectories.append(info.absoluteFilePath());
        qDebug() << "子目录名称：" << info.fileName(); // 获取子目录名称
    }
}
// 输出所有子目录路径
for(const QString& subDir : subDirectories){
    qDebug() << subDir;
}
```

### 自动滚动区QScrollArea

> 参考：https://blog.csdn.net/qq_31073871/article/details/83117430

QScrollArea属于控件容器类，可以直接在ui中拖出来。

当我们向QScrollArea里面拖入4个button时，可以发现，4个button并不是直接位于QScrollArea中的，而是位于它的成员scorllAreaWidgetContents中的，这个成员的类型也是控件类型QWidget，也就是说，QScrollArea这个容器本身就套了两层，我们放入的按钮等控件，都处在scrllAreaWidgetContents层，下文中我把QScrollArea.widget统一称之为“内部容器”或者"内容层"，内部容器是QScrollArea这个控件的子控件。

"内容层"相当于一块很大的幕布，按钮、label等控件都被绘制在了幕布上，而QScrollArea相当于一个小窗口，透过这个小窗口我们看一看到幕布上的一小部分内容，拖动滚动条相当于在窗口后面移动幕布，这样我们就能透过窗口看到幕布上不同位置的内容。

这个幕布本质上就是一个QWidget，如果QScrollArea是从UI设计师界面拖出来的，那么QT会自动为我们创建这个幕布，如果你是用代码new出来的QScrollArea，那么不要忘记同时new一个幕布widget，并通过QScrollArea::setWidget(QWidget *)把幕布和QScrollArea关联起来。

一句话总结何时出现滚动条：**只要幕布控件scorllAreaWidgetContents的大小超过了QScrollArea的大小，就会自动出现滚动条；如果幕布比观察窗口还小，那就不会出现滚动条**。一个简单方法是固定死scorllAreaWidgetContents有一个很大的高度，QScrollArea一个很小的高度，就能观察到滚动条。

### 裁剪图片

> 参考：https://blog.csdn.net/fanfanK/article/details/7260876

对于QImage，其scaled类的函数只能等比例改变大小，不能裁剪。要进行裁剪可以试用QImage的copy函数。例如，保留原图宽从0~300，高从0~250的部分，那么就：

```cpp
image=image.copy(0,0,300,250);
```

### QWidget 添加点击事件

> 参考：https://codeleading.com/article/54233987807/

在类定义中添加 clicked 信号， 重写 mouseReleaseEvent 函数

```cpp
#include <QWidget>
 
class MyWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MyWidget(QWidget *parent = nullptr);
 
signals:
    void clicked();
 
protected:
    virtual void mouseReleaseEvent(QMouseEvent * ev); 
 
};
```

在 mouseReleaseEvent 实现中发送 clicked 信号

```cpp
void MyWidget::mouseReleaseEvent(QMouseEvent * ev){
   emit clicked();
}
```

### 显示gif动画

> 参考：
>
> - https://blog.csdn.net/HeroGuo_JP/article/details/119637607
> - https://www.feiqueyun.cn/zixun/jishu/264951.html

Qt 中，静态图片 PNG，JPG 等可以用其创建 QPixmap，调用 QLabel::setPixmap() 来显示，但是能够具有动画的 GIF 却不能这么做，要在 QLabel 上显示 GIF，需要借助 QMovie 来实现。

使用 GIF 图片的路径创建 QMovie 对象，并且调用 QMovie::start() 启动 GIF 动画，然后通过 QLabel::setMovie() 设置好动画对象后，就能在 QLabel 上看到 GIF 动画了。

```cpp
#include <QApplication>
#include <QMovie>
#include <QLabel>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QLabel *label = new QLabel();
    QMovie *movie = new QMovie("/Users/Biao/Desktop/x.gif");
    label->setMovie(movie); // 1. 设置要显示的 GIF 动画图片
    movie->start();         // 2. 启动动画
    label->show();

    QObject::connect(movie, &QMovie::frameChanged, [=](int frameNumber) {
        // 控制 GIF 动画循环次数，GIF 动画执行一次就结束
        if (frameNumber == movie->frameCount() - 1) {
            movie->stop();
        }
    });

    return app.exec();
}

```

每次用户进入场景时都应播放动画。如果随着进入次数的增加，标签变大。可以使用：

```cpp
 mMovie->setScaledSize(this->size());  // mMovie class 为 QMovie
```

### QTabWiget的头部背景色设置

> 参考：https://www.cnblogs.com/bclshuai/p/11933912.html

QTabWiget的头部背景色通过设置background-color属性没有生效

解决办法：在Qt Designer中将autoFillBackground复选框勾选，设置背景色，就会自动填充颜色。

### 侧边栏隐藏和滑出（抽屉动画）

> 参考：https://zhuanlan.zhihu.com/p/647836590

界面控件很简单，主界面QWidget，侧边栏也用一个QWidget和一个按钮QPushbutton来进行组合。通过点击按钮来显示和隐藏侧边栏。主要用到的是控件的move()函数，配合QPropertyAnimation实现动画效果滑动显示隐藏。动画滑出动画效果使用到的是QPropertyAnimation类的setEasingCurve()函数，通过设置函数参数来实现不同的动画效果，具体效果可以通过Qt Create的帮助文件查询到。

```cpp
#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->widget_side->move(-ui->widget_side->width(),0);// 左侧停靠
    ui->pushButton->move(-1,ui->widget_side->height()/2);
    m_propertyAnimation = new QPropertyAnimation(ui->widget_side,"geometry");
    m_propertyAnimation->setEasingCurve(QEasingCurve::InOutSine);
    m_propertyAnimation->setDuration(800);
    m_propertyAnimation2 = new QPropertyAnimation(ui->pushButton,"geometry");
    m_propertyAnimation2->setEasingCurve(QEasingCurve::InOutSine);
    m_propertyAnimation2->setDuration(800);
}

MainWindow::~MainWindow()
{
    delete ui;
}void MainWindow::on_pushButton_clicked()
{
    //显示侧边栏
    if(!m_bSideflag)
    {
        m_propertyAnimation->setStartValue(QRect(-this->rect().width(),0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->setEndValue(QRect(0,0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->start();
        m_propertyAnimation2->setStartValue(QRect(-1,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->setEndValue(QRect(ui->widget_side->width()-2,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->start();
        ui->pushButton->setText("<<");
        m_bSideflag = !m_bSideflag;
    }
    else
    {
        m_propertyAnimation->setStartValue(QRect(0,0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->setEndValue(QRect(-this->rect().width(),0,ui->widget_side->width(),ui->widget_side->height()));
        m_propertyAnimation->start();
        m_propertyAnimation2->setStartValue(QRect(ui->widget_side->width()-2,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->setEndValue(QRect(-1,ui->widget_side->height()/2-ui->pushButton->height()/2,ui->pushButton->width(),ui->pushButton->height()));
        m_propertyAnimation2->start();
        ui->pushButton->setText(">>");
        m_bSideflag = !m_bSideflag;
    }
}
```

### QPushButton设置不可点击状态

> 参考：https://qt.0voice.com/?id=5989

使用setEnabled(False)方法

### 多选框与按钮组

最简单方法是从组件库中拖入一个多选框。

如果要对多选框的框和按钮实现更多控制，可以手动实现多选框。即选中几个按钮，右击，选择分配按钮组。当然已经组建好的按钮组也可以右键取消。

注意多选框逻辑：当选了之后，不能全部不选，如果要强制清空选项，使用代码：

```cpp
ui->buttonGroup->setExclusive(false);
ui->buttonGroup->checkedButton()->setChecked(false);
ui->buttonGroup->setExclusive(true);
```

