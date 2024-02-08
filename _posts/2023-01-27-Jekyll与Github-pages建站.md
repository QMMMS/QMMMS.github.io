---
title: Jekyll与Github Pages建站
date: 2023-01-27 14:50:00 +0800
categories: [技术]
tags: [github]

mermaid: true
---

## 说明

---------

在博主学习完**markdown**，**git**，**Linux**，**计算机网络**以及**HTML**，**CSS**，**JavaScript**三件套基础后，又想折腾了。如果想要搭建一个个人博客，这是最好的时机。

下面记录了使用Jekyll与GitHub pages建站的过程。

一些有用的网站如下：

-   GitHub pages官网：https://pages.github.com/
-   Jekyll官网：https://jekyllrb.com/
-   Jekyll官网文档：https://jekyllrb.com/docs/
-   GitHub Pages官方文档：https://docs.github.com/zh/pages
-   Jekyll - 静态网站生成器教程双语字幕B站视频：https://www.bilibili.com/video/BV1qs41157ZZ/?p=1
-   Jekyll主题网站：http://jekyllthemes.org/

## 创建GitHub pages

------

搭建自己的博客，朴素的想法是租借一台服务器从零配置。这里使用GitHub pages，优点包括：

-   免费
-   即使不懂技术细节，也只需按步骤一步步操作即可
-   支持的功能多，玩法丰富，你可以绑定你的域名、使用免费的 HTTPS、自己 DIY 网站的主题、使用他人开发好的插件等等
-   当完成搭建后，只需要专注于文章创作，其他诸如环境搭建、系统维护、文件存储的事情一概不用操心，都由 GitHub 处理

创建流程：

1.   默认你已经拥有**GitHub**账号
2.   创建一个新的 public repository， 名字为 `[你的GitHub账号名].github.io`，这也是你网站的初始网址
3.   将仓库克隆到本地，添加一个`index.html`，里面随便写点什么，push到远程仓库
4.   浏览器访问`[你的GitHub账号名].github.io`，你能看到你写的`index.html`啦！

>   GitHub pages官网以及简单流程：https://pages.github.com/
>
>   某个GitHub Pages 搭建教程：https://sspai.com/post/54608

## 安装Jekyll

------

你已经拥有自己的网站了，朴素的想法是从零开始编写HTML，CSS与JavaScript文件。~~但是我太菜了~~

如果想快速实现好看的页面呢？可以考虑使用**Jekyll**，这也是GitHub官方推荐的。

>   Jekyll 是一个静态站点生成器，内置 GitHub Pages 支持和简化的构建过程。 Jekyll 使用 Markdown 和 HTML 文件，并根据您选择的布局创建完整静态网站。
>
>   来自GitHub文档：https://docs.github.com/zh/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll

了解了Jekyll之后，让我们正式开始吧。

>   注意，博主的电脑为Mac，以下内容来源：
>
>   -   Jekyll官网文档：https://jekyllrb.com/docs/
>   -   Mac安装Jekyll教程：https://blog.csdn.net/wanghao_sh/article/details/128196126

Jekyll 的本质是一个Ruby Gem组件，使用Ruby编写，所以需要相关依赖。我们首先需要：

-   [GCC](https://gcc.gnu.org/install/)。执行 `gcc -v`,`g++ -v` 检查
-   [Make](https://www.gnu.org/software/make/)。执行`make -v` 检查
-   [Ruby](https://www.ruby-lang.org/en/downloads/)  **2.5.0** 或更高版本。执行`ruby -v`检查
-   [RubyGems](https://rubygems.org/pages/download) 。执行`gem -v`检查

>   对Ruby一窍不通？看看这两个介绍视频/文档：
>
>   -   100秒介绍Ruby：https://www.bilibili.com/video/BV1PU4y1z7Fs/
>   -   Ruby 101：https://jekyllrb.com/docs/ruby-101/

### 安装Homebrew

为了安装**Ruby**，在Mac上首先推荐安装**Homebrew**。Homebrew是一款Mac OS平台下的软件包管理工具，安装软件而不用关心各种依赖和文件路径的情况，十分方便快捷。

在使用brew安装软件时, 经常会报错 failed to connect to [http://raw.githubusercontent.com](https://link.zhihu.com/?target=http%3A//raw.githubusercontent.com) port 443, 出现的原因是DNS污染，推荐Mac安装brew的命令：

```shell
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```

>   原文：https://zhuanlan.zhihu.com/p/111014448

>   几个brew命令
>
>   -   本地软件库列表：`brew ls`
>   -   查找软件：`brew search [关键字]`
>   -   查看brew版本：`brew -v `
>   -   更新brew版本：`brew update`
>   -   安装cask软件：`brew install --cask [软件]`

使用命令安装完brew后显示warning：

```
Warning: No remote 'origin' in /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask, skipping update and reset!
Warning: No remote 'origin' in /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core, skipping update and reset!
Warning: No remote 'origin' in /usr/local/Homebrew/Library/Taps/homebrew/homebrew-services, skipping update and reset!
```

不过后面马上有解决方法，把命令运行一下就行：

```
fatal: unsafe repository ('/usr/local/Homebrew/Library/Taps/homebrew/homebrew-core' is owned by someone else)
To add an exception for this directory, call:

	git config --global --add safe.directory /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core
Homebrew/homebrew-core (no Git repository)
```

### 安装Ruby

安装：

```shell
brew install ruby
```

切换版本与设置，可能要使用管理员权限：

```shell
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.zshrc
export LDFLAGS="-L/usr/local/opt/ruby/lib"
export CPPFLAGS="-I/usr/local/opt/ruby/include"
```

重进一下终端，查看Ruby版本：

```shell
ruby -v
```

显示`ruby 3.2.0 (2022-12-25 revision a528908271) [x86_64-darwin21]`，安装完成。

### 安装与配置Jekyll

1. 安装Jekyll：

```shell
gem install jekyll
```

2. 安装bundler（功能类似Make）：

```shell
gem install jekyll bundler
```

3. 在 `./myblog` 目录下创建一个全新的Jekyll网站：

```shell
jekyll new myblog
```

4. 进入新创建的目录：

```shell
cd myblog
```

5. 构建网站并启动一个本地web服务：

```shell
 bundle exec jekyll serve
```

6.   在浏览器上打开网址`http://127.0.0.1:4000/`，你能看到初始的网页啦！
7.   命令行按Ctrl+c关闭服务器

## 探索默认主题

------

让我们探索一下Jekyll以及提供好的默认网页主题，你可以看看`./myblog`文件夹下Jekyll做了什么，初始的文件结构是这样的：

```
.
├── 404.html
├── Gemfile
├── Gemfile.lock
├── _config.yml
├── _posts
│   └── 2023-01-24-welcome-to-jekyll.markdown
├── _site
│   ├── 404.html
│   ├── about
│   │   └── index.html
│   ├── assets
│   │   ├── main.css
│   │   ├── main.css.map
│   │   └── minima-social-icons.svg
│   ├── feed.xml
│   ├── index.html
│   └── jekyll
│       └── update
│           └── 2023
│               └── 01
│                   └── 24
│                       └── welcome-to-jekyll.html
├── about.markdown
└── index.markdown
```

其中，`_site`文件夹是最终生成的网页，我们一般不去修改里面的内容。

>   **探索默认网页主题**的内容大多来自B站视频**Jekyll - 静态网站生成器教程双语字幕**：https://www.bilibili.com/video/BV1qs41157ZZ/?p=1

### 写博客

把你的博客内容放在`_posts`文件夹下面，注意文件格式，具体参考Jekyll已经写好的`welcome-to-jekyll.markdown`

其中，`.md`开头会有三条下划线包裹的内容，这定义了博文的重要消息，比如：

```yaml
---
layout: post
title:  "Welcome to Jekyll!"
date:   2023-01-24 14:40:52 +0800
categories: jekyll update
---
```

当你写好博文，再次运行服务器：

```
jekyll serve
```

看看你新写的博客是否出现在了网站上！

### 更改全局设置

`_config.yml` 文件保存全局设置，包括标题，描述，构建设置。使用YAML 语法。具体参考Jekyll已经写好的`_config.yml` 文件。

>   YAML 是一种配置语言，语法参考：
>
>   -   https://learn-the-web.algonquindesign.ca/topics/markdown-yaml-cheat-sheet/#yaml
>   -   https://learnxinyminutes.com/docs/yaml/

### 自定义其他内容

-   修改**关于（about）**页面：`about.markdown`
-   修改**主页（home）**：`index.markdown`
-   配置依赖：`Gemfile`

### 草稿

在`./myblog`文件夹下新建一个文件夹`_drafts`，在里面可以放置一些草稿`.md`文件。当你使用普通的命令`jekyll serve`时，草稿不会出现在你的博客上，如果使用`jekyll serve --draft`命令，草稿会出现在你的博客上。

### 创建页面

如果你想为你的博客网站添加一个普通页面，而不是博文，应该怎么做？如果你想在`http://127.0.0.1:4000/others/donate`创建一个捐赠页面，直接在`./myblog`文件夹下新建一个文件夹`others`，在里面可以放置一个`donate.md`文件，内容可以是：

```markdown
---
layout: page
title: Donate
---

please donate me.
```

使用命令`jekyll serve`，看看`http://127.0.0.1:4000/others/donate`吧！

你可以感受到这种一一对应的关系，jekyll会自动构建这些普通文件夹和文件，默认情况下，Jekyll 不会构建以下文件或文件夹：

-   位于名为 `/node_modules` 或 `/vendor` 的文件夹中
-   以 `_`、`.` 或 `#` 开头
-   以 `~` 结尾
-   被配置文件中的 `exclude` 设置排除

如果想要 Jekyll 处理其中任何文件，可以使用配置文件中的 `include` 设置。

### 永久链接

让我们观察一下我们的博文最终位置在哪，对于如下的头信息：

```yaml
---
layout: post
title:  "Welcome to Jekyll!"
date:   2023-01-24 14:40:52 +0800
categories: jekyll update
---
```

我们看到博文的最终链接为`http://127.0.0.1:4000/jekyll/update/2023/01/24/welcome-to-jekyll.html`

如果我们不想jekyll自动帮我们分配这个链接，可以在头信息加上一行：

```yaml
permalink: "/welcome/binginer.html"
```

最终的链接为：`http://127.0.0.1:4000/welcome/binginer.html`，这样手动确定是推荐的

### 自定义布局

注意到头信息中有这么一行：

```yaml
layout: post
```

布局是内容的模板，这行的意思是使用名为`post`的布局，初始`post`布局是默认提供并且写死的。

那么，如何创建自己的布局？如何重写默认布局？在这小节中我们尝试重写`post`布局。

1.   在`./myblog`文件夹下新建一个文件夹`_layouts`

2.   在文件夹`_layouts`中创建文件`wrapper.html`，我们意图使用`wrapper.html`作为其他模板的基础**HTML**文件

3.   编写文件`wrapper.html`，内容如下：
{% raw %}
     ```html
     <html>
         <head>
             <meta charset="UTF-8">
             <title>{{ site.title }}</title>
         </head>
         <body>
             {{ content }}
         </body>
     </html>
     ```
{% endraw %}
     解释一下，之所以称之为**模板**，很多内容不是固定的，特别是使用双大括号包裹的部分：

     -   {% raw %}`{{ site.title }}`{% endraw %}将读取`_config.yml` 文件中网站的标题
     -   {% raw %}`{{ content }}`{% endraw %}将替换为使用本模板的文件内容

4.   在文件夹`_layouts`中创建文件`post.html`，我们意图使用`post.html`重写默认的布局，并且`post.html`将基于`wrapper.html`

5.   编写文件`post.html`，内容如下：
{% raw %}
     ```html
     ---
     layout: wrapper
     ---

     <h1>{{ page.title }}</h1>
     <hr>

     {{ content }}
     ```
{% endraw %}
6.   重新打开你的网站，看看我们编写的`post`布局是否覆盖了默认布局！

>   布局中其他可以使用的变量，详见官方文档：https://jekyllrb.com/docs/variables/

### 使用include与参数

这一小节介绍使用使用include与参数更加灵活地编写模板上的组件，比如header，footer等。

1.   在`./myblog`文件夹下新建一个文件夹`_includes`

2.   在新文件夹`_includes中`创建文件`header.html`，我们意图使`header.html`成为每个帖子的头部

3.   编写文件`header.html`，内容如下：
{% raw %}
     ```html
     <h3 style="color: {{ include.color }};">{{ site.title }}</h3>

     <hr>
     ```
{% endraw %}
4.   修改上文我们提到的文件`wrapper.html`，内容如下：
{% raw %}
     ```html
     <html>
         <head>
             <meta charset="UTF-8">
             <title>{{ site.title }}</title>
         </head>
         <body>
             {% include header.html color="blue" %}
             {{ content }}
         </body>
     </html>
     ```
{% endraw %}
5.   重新打开你的网站，看看我们编写的`header.html`是否成为了每个帖子的头部！

6.   解释：{% raw %}`{% include header.html color="blue" %}`{% endraw %}引入了`header.html`文件，并且把参数`color`传到`header.html`文件中，非常灵活

### 其他模版语句

类似于上面的include语句，我们可以在html模板中加入更多语句：

-   循环
{% raw %}
    ```html
    {% for post in site.posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
    ```
{% endraw %}
-   条件
{% raw %}
    ```html
    {% if page.title == "Welcome to Jekyll!" or page.title == "...."%}
      <strong>嘿！这是个欢迎界面！</strong>
    {% elsif page.title == "Jekyll与GitHub pages建站" %}
      <strong>嘿！这是个教程页面！</strong>
    {% else %}
      <strong>普通页面</strong>
    {% endif %}
    ```
{% endraw %}
{% raw %}
    ```html
    {% for post in site.posts %}
      <li><a style="{% if page.url == post.url %}color: red;{% endif %}" href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
    ```
{% endraw %}
### 数据文件

我们如何存储数据？访问数据？这小节介绍Jekyll的数据文件。

1.   在`./myblog`文件夹下新建一个文件夹`_data`

2.   在新文件夹`_data中`创建文件`people.yml`，我们意图使用`people.yml`存储一些人的信息

3.   编写文件`people.yml`，内容如下：

     ```yaml
     - name : "QMS"
       ocupation : "博主"

     - name : "SMQ"
       ocupation : "搬运工"

     - name : "MSQ"
       ocupation : "摸鱼"
     ```

4.   我们可以在html模板中加入这些语句来访问数据：
{% raw %}
     ```html
     {% for person in site.data.people %}
       {{ person.name }}, {{ person.ocupation }} <br>
     {% endfor %}
     ```
{% endraw %}
### 静态文件

你也许会在网站上放置除了HTML的其他文件，比如图片，PDF文档。建议使用`./assets/img`等文件夹来规范，但这不是必须的，Jekyll会自动找到它们，也提供了一些语句来获取它们，比如：

-   {% raw %}`{% for file in site.static_files %}`{% endraw %}
-   {% raw %}`{{ file.basename }}`{% endraw %}
-   {% raw %}`{{ file.path }}`{% endraw %}
-   {% raw %}`{{ file.name }}`{% endraw %}
-   等等。。。

>   一些简单的介绍到此为止了，如果想继续探索，看看官方文档：https://jekyllrb.com/docs/

## 在GitHub Pages上部署

------

之前我们都在本地部署，现在我们把这些成品放到GitHub Pages上。

方法一是将所有文件放到你的`[你的GitHub账号名].github.io`仓库上，GitHub会识别这个仓库名并且自动帮你构建。

方法二不要求特定的仓库名，在一个普通仓库上创建`gh-pages`分支，将所有文件放到上面，GitHub会识别这个分支名并且自动帮你构建。

注意到方法二可以使用多个Jekyll组件，构建一个网站群，如果想要在二级目录上构建，或者说希望主页是`XXX.github.io/blog`，填写`_config.yml` 文件中的`baseurl`。

注意，如果使用`baseurl`，之前提到的`post.url`和`page.url`是不包括这部分的，需要用`site.baseurl`引入。

## 使用炫酷主题

------

是否觉得默认主题太无趣了？但自己又写不出来？试试他人编写的炫酷主题！

首先找到喜欢的主题，一些Jekyll主题网站：

-   Jekyll主题网站：http://jekyllthemes.org/
-   Jekyll官网的主题：https://jekyllrb.com/showcase/
-   Github流行统计：https://github.com/topics/jekyll-template

在这一节中，博主使用主题Chirpy来搭建博客。

-   官方demo：https://chirpy.cotes.page/
-   github仓库：https://github.com/cotes2020/jekyll-theme-chirpy
-   其他博主使用Chirpy的经验：https://whuwangyong.github.io/2022-03-29-jekyll/#%E5%AE%89%E8%A3%85ruby%E5%92%8Cjekyll

以下大部分内容来自上面的链接。

### 设置仓库与部署

1.   使用Chirpy Starter(https://github.com/cotes2020/chirpy-starter/generate)来创建仓库，并且改名为`[你的GitHub账号名].github.io`

2.   克隆到本地，使用`bundle`命令安装依赖

     >   使用`bundle`命令时报错，需要低版本的`ruby`，使用旧版本步骤：
     >
     >   1.   使用命令`brew search ruby`看看有没有旧版本，在列表中看到`ruby@3.1`
     >   2.   安装命令`brew install ruby@3.1`
     >   3.   和最初安装时一样，切换版本与设置，安装时会提示。

3.   在本地运行与测试，使用命令`bundle exec jekyll s`

4.   看看http://localhost:4000是否出现了页面

5.   如果不是linux系统构建的本地服务器，为了与GitHub Pages的linux环境一致，使用命令：

     ```shell
      bundle lock --add-platform x86_64-linux
     ```

6.   修改`.github/workflows/pages-deploy.yml`文件，`on.push.branches`应该与GitHub上的分支名一致

7.   GitHub的仓库页面上. 在 *Settings* 页中点击 *Pages* 。之后在 **Source** 选择 **GitHub Actions**

8.   push到远程仓库，**GitHub Actions**会在一次push之后自动开始构建与部署

9.   GitHub的仓库页面上的 *Actions* 页中，等待一切完成

10.   浏览器访问`[你的GitHub账号名].github.io`，看看你的博客吧！

### 更改全局设置

照例修改`_config.yml` 文件，包括网站名，更改语言，时区，头像，简介，开启评论等等，每个设置都有注释解释，照做就行。几个难理解的如下：

-   `avatar: assets/img/avatar/wy-avatar-339x335.jpg # 头像`

-   `google_site_verification: googledxxxxxxx.html # 去Google Search Console申请，用于google收录`

-   注意，开启评论后，记得在你的网站里使用一次评论，第一次会失败，但是会自动引导你为仓库安装utterances（utterances详见：https://github.com/utterance）

    ```yaml
    comments:
      active: utterances # 使用github issue作为文章的评论系统
      utterances:
        repo: QMMMS/QMMMS.github.io        # <gh-username>/<repo>
        issue_term: title  # < url | pathname | title | ...>
    ```

### 其他设置

-   配置文章的分享选项，如Facebook、微博，修改`_data/share.yml`
-   配置文章的左下角其他站点链接，修改`_data/contact.yml`
-   自定义网站图标，将所有图标文件放入`assets/img/favicons/`，具体详见https://chirpy.cotes.page/posts/customize-the-favicon/
-   自定义“关于我”的内容，修改`_tabs/about.md`

### 写博客

与之前一样，博客文件名必须是`YYYY-MM-DD-TITLE.md`，放在`_posts`文件夹。

头信息可以是：

```yaml
---
title: TITLE
date: YYYY-MM-DD HH:MM:SS +/-TTTT # 2022-01-01 13:14:15 +0800 只写日期也行；不写秒也行；这样也行 2022-03-09T00:55:42+08:00
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase

# 以下内容都有默认值，可以不写


# 以下默认false
math: true
mermaid: true
pin: true

# 以下默认ture
toc: false
comments: false
---
```

-   其中`math: true`可以编辑数学公式

-   **mermaid**美人鱼，是一个类似 markdown，用文本语法来描述文档图形 (流程图、 时序图、甘特图) 的工具，例如：

    ```mermaid
    graph TD;
        A-->B;
        A-->C;
        B-->D;
        C-->D;
    ```

    ```mermaid
    gantt
        title  Adding GANTT diagram functionality to mermaid
        apple :a, 2017-07-20, 1w
        banana :crit, b, 2017-07-23, 1d
        cherry :active, c, after b a, 1d
    ```

-   **categories**支持多层级。如`categories: [Java, Spring, AOP]`，最终的分类效果是`Java/Spring/AOP`，这样就可以复用笔记软件里面设置好的分类

-   `toc: false`收起页面右侧的大纲

图片必须放在`/assets/`下。最佳实践：放在`/assets/img/posts/[YYYY-MM-DD-TITLE]`目录下，可以在头信息中加入比如`img_path: "/assets/img/posts/2023-01-26-测试"`，之后引用图片只需要文件名，路径会自动加入，比如`![码农meme](rabbyte.jpg)`

带样式的框框如下：

```markdown
> 信息
{: .prompt-info }

> 提示
{: .prompt-tip }

> 警告
{: .prompt-warning }

> 危险
{: .prompt-danger }
```

TODO list：

```
- [ ] Job
  + [x] Step 1
  + [x] Step 2
  + [ ] Step 3
```

### 更改作者

如果要更改博客的作者或者添加多个作者，如下操作：

1.   创建文件`_data/authors.yml`，内容格式如下：

     ```yaml
     <author_id>:
       name: <full name>
       twitter: <twitter_of_author>
       url: <homepage_of_author>
     ```

2.   在博客头信息加入如下信息：

     ```
     ---
     author: <author_id>                     # 一个作者
     # 或者
     authors: [<author1_id>, <author2_id>]   # 多个作者
     ---
     ```

### 添加左侧栏tab

如果新增“友情链接”tab。在`_tabs`目录下新建`links.md`:

```markdown
---
title: 友情链接
icon: fas fa-link
order: 5
---
```

调整order和icon。icon去[Font Awesome Icons](http://www.fontawesome.com.cn/)找。然后修改`_data/locales/en.yml`和`_data/locales/zh-CN.yml`，在`tabs:`下添加`links: Links`和`links: 友链`，以适配中英文。


