---
title: Vue+SpringBoot实现大创管理系统
date: 2024-05-25 12:21:00 +0800

media_subpath: "/assets/img/posts/2024-05-25-VueSpringboot"
categories: [经验与总结]
tags: [经验]
---

> - [项目仓库](https://gitee.com/QMMMS/ipmsfcsv3)
> - [参考教程](https://www.bilibili.com/video/BV1nV4y1s7ZN)

## 实现界面

![](sv3.png)

![](sv1.png)

![](sv2.png)

![](sv4.png)

## 技术总览

工具：

- 后端开发：IDEA
- 包管理与构建：Maven
- 前端开发：VSCode 
- 接口调试：ApiPost

后端：

- 框架：SpringBoot
- 设计风格：RESTful
- 操作数据库：MyBatis-Plus
- 数据库：MySQL
- 身份认证：JWT 

前端：

- 框架：Vue
- 页面组件：ElementUI+FontAwesome
- 网络请求：Axios
- 页面路由：VueRouter
- 状态管理：Vuex
- 数据模拟：MockJS 

部署：

- 云服务器：华为云
- Web服务器：Nginx

## Maven

> 参考：[Maven安装和配置&详细步骤](https://blog.csdn.net/weixin_56800176/article/details/127949796)

在原生态代码无框架的时候，我们最痛苦的一件事就是导入各种各样的jar包，jar包太多以至于我们很难管理，项目功能稍多，就会出现好多好多的包，你要考虑在哪找这个包，还有它的包的依赖，让人很痛苦！

Maven是一个流行的Java项目管理工具，它提供了一种结构化的方式来管理Java项目的构建、依赖、文档和发布。Maven基于项目对象模型（Project Object Model，POM）来管理项目，通过定义一系列规范化的目录结构和配置文件来管理项目的构建过程和依赖关系。Maven的主要作用是提高Java项目的可维护性、可重用性和可扩展性。

运行 Maven 的时候，Maven 所需要的任何构件都是直接从本地仓库获取的。如果本地仓库没有，它会首先尝试从远程仓库下载构件至本地仓库。

![](maven.png)

主要用途：

- 项目构建：提供标准的，跨平台的自动化构建项目的方式
- 依赖管理：方便快捷的管理项目依赖的资源（jar包），避免资源间的版本冲突等问题
- 统一开发结构：提供标准的，统一的项目开发结构

在[官方下载界面](https://maven.apache.org/download.cgi)选择Binary zip archive，进行解压，”然后在系统环境中配置环境变量，在path中配置到bin目录。

查看maven是否安装配置完成：

```
PS C:\Users\15951> mvn -version
Apache Maven 3.9.2 (c9616018c7a021c1c39be70fb2843d6f5f9b8a1c)
Maven home: C:\Program Files\apache-maven-3.9.2
Java version: 20.0.1, vendor: Oracle Corporation, runtime: C:\Program Files\Java\jdk-20
Default locale: zh_CN, platform encoding: UTF-8
OS name: "windows 11", version: "10.0", arch: "amd64", family: "windows"
```

设置本地仓库地址，找到 conf 文件夹下的 settings.xml 文件进行修改：

```xml
<!-- localRepository
| The path to the local repository maven will use to store artifacts.
|
| Default: ${user.home}/.m2/repository
<localRepository>/path/to/local/repo</localRepository>
自己选择一个本地文件夹
-->
<localRepository>D:\Data\Maven</localRepository>
```

配置国内镜像：

```xml
<mirror>   <!-- 配置阿里云镜像仓库 -->
    <id>alimaven</id>
    <name>aliyun maven</name>
    <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
    <mirrorOf>central</mirrorOf>
</mirror>
```

在IDEA中配置：

设置->构建、执行、部署->构建工具->Maven

修改主路径、用户设置文件。

**安装依赖例子**，创建SpringBoot项目勾选Spring Web选项后，会自动将spring-boot-starter-web组件加入到项目中。 Spring Boot将传统Web开发的mvc、json、tomcat等框架整合，提供了spring-boot-starter-web组件，简化了Web应用配置。

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
```

## SpringBoot

Spring Boot是基于Spring的框架，该框架使用了特定的方式来进行配置，从而使开发人员不再需要定义样板化的配置。简化开发过程。Spring Boot是所有基于Spring开发项目的起点。

Spring Boot集成了绝大部分目前流行的开发框架，就像Maven集成了所有的JAR包一样，Spring Boot集成了几乎所有的框架，使得开发者能快速搭建Spring项目。

SpringBoot特点：

- 遵循“约定优于配置”的原则，只需要很少的配置或使用默认的配置。
- 能够使用内嵌的Tomcat、Jetty服务器，不需要部署war文件。
- 提供定制化的启动器Starters，简化Maven配置，开箱即用。
- 纯Java配置，没有代码生成，也不需要XML配置。 
- 提供了生产级的服务监控方案，如安全监控、应用监控、健康检测等。

### 安装和配置

Spring Boot不用手动安装，在IDEA完整版新建项目，选择**Spring Initializr**：

- 服务器URL可以填https://start.springboot.io，国内的
- 依赖项在Web中找到Spring Web

### 目录结构

![](sbdir.png)

项目创建成功后会默认在resources目录下生成`application.properties`文件。该文件包含Spring Boot项目的全局配置。 配置格式如下：

```
server.port=8080
```

### Controller

1. 创建子目录controller 
2. 在目录controller中，创建HelloController.java文件
3. 启动项目，在浏览器窗口中输入`http://localhost:8080/hello`

```java
@RestController
public class HelloController {
    // http://localhost:8080/hello	
    //注解@RequestMapping用来映射请求的URL和请求的方法等。本例用来映射"/hello"的请求，请求方法为get
    @RequestMapping(value = "/hello", method = RequestMethod.GET)
    //等价于@GetMapping("/hello")
    public String hello() {
        return "你好世界！！！";
    }
}
```

Spring Boot提供了@Controller和@RestController两种注解来标识此类负责接收和处理HTTP请求。 

- 如果请求的是页面和数据，使用@Controller注解即可
- 如果只是请求数据，则可以使用@RestController注解。默认情况下，@RestController注解会将返回的对象数据转换为JSON格式（例如返回 User 类）

![](spcon.png)

### Interceptor

拦截器在Web系统中非常常见，对于某些全局统一的操作，我们可以把它提取到拦截器中实现。总结起来，拦截器大致有以下几种使用场景：

- 权限检查：如登录检测，进入处理程序检测是否登录，如果没有，则直接返回登录页面。
- 性能监控：有时系统在某段时间莫名其妙很慢，可以通过拦截器在进入处理程序之前记录开始时间，在处理完后记录结束时间，从而得到该请求的处理时间
- 通用行为：读取cookie得到用户信息并将用户对象放入请求，从而方便后续流程使用，还有提取Locale、Theme信息等，只要是多个处理程序都需要的，即可使用拦截器实现。

Spring Boot定义了HandlerInterceptor接口来实现自定义拦截器的功能HandlerInterceptor 接口定义了preHandle、postHandle、afterCompletion三种方法，通过重写这三种方法实现请求前、请求后等操作

![](spic.png)

```java
public class LoginInterceptor implements HandlerInterceptor{
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        System.out.println("LoginInterceptor.preHandle");
        return true; // 可以编写 if 语句在特定情况下return false拦截
    }
}
```

### Config

拦截器注册

- addPathPatterns方法定义拦截的地址 
- excludePathPatterns定义排除某些地址不被拦截 
- 添加的一个拦截器没有addPathPattern任何一个url则默认拦截所有请求
- 如果没有excludePathPatterns任何一个请求，则默认不放过任何一个请求。

```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new LoginInterceptor()).addPathPatterns("/user/**");
    }
}
```

## RESTful

RESTful是目前流行的互联网软件服务架构设计风格。REST并不是一个标准，它更像一组客户端和服务端交互时的架构理念和设计原则，基于这种架构理念和设计原则的Web API更加简洁，更有层次。

- 每一个URI代表一种资源 
- 客户端使用GET、POST、PUT、DELETE四种表示操作方式的动词对服务端资源进行操作：GET用于获取资源，POST用于新建资源（也可以用于更新资源），PUT用于更新资源，DELETE用于删除资源。 
- 通过操作资源的表现形式来实现服务端请求操作。 
- 资源的表现形式是JSON或者HTML。 n 客户端与服务端之间的交互在请求之间是无状态的，从客户端到服务端的每个请求都包含必需的信息。

符合RESTful规范的Web API需要具备如下两个关键特性：

- 安全性：安全的方法被期望不会产生任何副作用，当我们使用GET操作获取资源时，不会引起资源本身的改变，也不会引起服务器状态的改变。
- 幂等性：幂等的方法保证了重复进行一个请求和一次请求的效果相同（并不是指响应总是相同的，而是指服务器上资源的状态从第一次请求后就不再改变了），在数学上幂等性是指N次变换和一次变换相同。

状态码分为以下5个类别：

- 1xx：信息，通信传输协议级信息
- 2xx：成功，表示客户端的请求已成功
- 3xx：重定向，表示客户端必须执行一些其他操作才能完成其请求
- 4xx：客户端错误，此类错误状态码指向客户端
- 5xx：服务器错误，服务器负责这写错误状态码

> 可以使用Swagger工具，能够自动生成完善的RESTful API文档，同时并根据后台代码的修改同步更新，同时提供完整的测试页面来调试API。

## MyBatis-Plus

ORM（Object Relational Mapping，对象关系映射）是为了解决面向对象与关系数据库存在的互不匹配现象的一种技术。 ORM通过使用描述对象和数据库之间映射的元数据将程序中的对象自动持久化到关系数据库中。 ORM框架的本质是简化编程中操作数据库的编码。

![](orm.png)

MyBatis能够非常灵活地实现动态SQL，可以使用XML或注解来配置和映射原生信息，能够轻松地将Java的POJO（Plain Ordinary Java Object，普通的Java对象）与数据库中的表和字段进行映射关联。 

MyBatis-Plus是一个 MyBatis 的增强工具，在 MyBatis 的基础上做了增强，简化了开发。在Maven配置项添加依赖，配置好propoties数据库信息就能用。

**目录结构：**

![](myb.png)

可以看到新内容是`mapper`文件夹，一个类的基本结构如下：

```java
@Mapper
public interface StudentMapper extends BaseMapper<Student> {    
    @Select("select * from student where ID=#{ID}")
    Student selectById(String ID);
}
```

其他CRUD操作示例：

```java
@Insert("insert into student values(#{ID},#{name},#{deptName},#{totCred})")
public int insert(Student student);

@Delete("delete from student where ID=#{ID}")
public int delete(String ID);

@Update("update student set tot_cred=#{totCred} where ID=#{ID}")
public int updateTotCred(int totCred, String ID);
```

实现复杂关系映射，可以使用@Results注解：

```java
@Select("select * from student")
@Results(
    {
        @Result(column = "ID",property = "ID"),
        // column是数据库字段名，property是实体类属性名
        @Result(column = "name",property = "name"),
        @Result(column = "dept_name",property = "deptName"),
        @Result(column = "tot_cred",property = "totCred"),
        // 对于学生的 takes 属性，使用 ID 字段的值作为参数，调用 selectByStudentId 方法，执行关联查询，
        // 获取学生的选课信息，并将结果映射到 takes 属性（类型为 List）。
        @Result(column = "ID",property = "takes",javaType = List.class,
                many = @Many(select = "com.example.demo.mapper.TakeMapper.selectByStudentId"))
    }
)
List<Student> selectAllStudentAndTakes();
```

在Controller中使用：

```java
@Autowired
private StudentMapper studentMapper;

@GetMapping("/student")
public List query(){
    List<Student> list = studentMapper.selectList(null); // select * from student
    System.out.println(list);
    return list;
}

@PostMapping("/student/findById")
public Student findById(String ID){
    return studentMapper.selectById(ID);// select * from student where ID=#{ID}
}
```

## ApiPost

ApiPost是一款支持模拟POST、GET、PUT等常见HTTP请求,支持团队协作,并可直接生成并导出接口文档的API 文档、调试、Mock、测试一体化协作性能非常强大的工具。简单说：ApiPost = Postman + Swagger + Mock。

ApiPost产生的初衷是为了提高研发团队各个角色的效率！产品的使用受众为由前端开发、后端开发和测试人员以及技术经理组成的整个研发技术团队。ApiPost通过协作功能将研发团队的每个角色整合打通。

安装：[官网](https://www.apipost.cn/)免费安装，选择Windows

之后Post请求的测试可以使用ApiPost来测试。

## Vue

Vue是一种强大的JavaScript框架,用于构建现代化的Web应用程序。它具有组件化架构、响应式编程、强大的工具和插件以及良好的生态系统。

### 安装

> 需要先安装nodejs，安装过程这里不再列出。
>
> - [什么是Node.js](https://www.jb51.net/article/210601.htm)：让 JavaScript 运行在服务器上。
> - [什么是npm](https://blog.csdn.net/csdn_jwdlh/article/details/124093999)：JavaScript 世界的包管理工具,并且是 Node.js 平台的默认包管理工具。

安装 Vue 官方的项目脚手架工具：

```
$ npm init vue@latest
Need to install the following packages:
  create-vue@3.6.1
Ok to proceed? (y) y

Vue.js - The Progressive JavaScript Framework
# 这里需要进行一些配置，项目名输入 runoob-vue3-test，其他默认回车即可
&#x2714; Project name: … runoob-vue3-test
&#x2714; Add TypeScript? … No / Yes
&#x2714; Add JSX Support? … No / Yes
&#x2714; Add Vue Router for Single Page Application development? … No / Yes
&#x2714; Add Pinia for state management? … No / Yes
&#x2714; Add Vitest for Unit Testing? … No / Yes
&#x2714; Add an End-to-End Testing Solution? › No
&#x2714; Add ESLint for code quality? … No / Yes

Scaffolding project in /Users/tianqixin/runoob-test/runoob-vue3/runoob-vue3-test...

Done. Now run:

  cd runoob-vue3-test
  npm install
  npm run dev
```

在项目被创建后，通过以下步骤安装依赖并启动开发服务器：

```
$ cd runoob-vue3-test
$ npm install
$ npm run dev
  VITE v4.3.4  ready in 543 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h to show help
```

成功执行以上命令后访问 **http://localhost:5173/**，可以看到网页啦！

> - `npm install ......`是局部安装，安装后的东西只保留在当前项目中
>- `npm install -g .......`是全局安装，配置好环境变量之后可以在控制台中访问

**另外一种创建项目方式**

- 全局安装vue-cli：`npm install -g @vue/cli`
- 查看是否安装完成：`vue -V`
- 创建项目：`vue create <项目名称>`
- 运行项目：`npm run serve`
- 上传到远程仓库上是不带本地node依赖的，如果删掉了依赖要重新下载，使用命令：`npm install`

### 介绍

Vue 是一套用于构建用户界面的渐进式框架。Vue.js提供了MVVM数据绑定和一个可组合的组件系统，具有简单、灵活的API。其目标是通过尽可能简单的API实现响应式的数据绑定和可组合的视图组件

> MVVM是Model-View-ViewModel的缩写，它是一种基于前端开发的架构模式，其核心是提供对View和ViewModel的双向数据绑定。
>
> Vue提供了MVVM风格的双向数据绑定，核心是MVVM中的VM，也就是ViewModel，ViewModel负责连接View和Model，保证视图和数据的一致性。

![](vuemvvm.png)

组件（Component）是Vue.js最强大的功能之一。组件可以扩展HTML元素，封装可重用的代码。Vue的组件系统允许我们使用小型、独立和通常可复用的组件构建大型应用。

Vue 中规定组件的后缀名是 .vue，每个 .vue 组件都由 3 部分构成，分别是：

- template，组件的模板结构，可以包含HTML标签及其他的组件
- script，组件的 JavaScript 代码
- style，组件的样式

下图为开发界面和项目结构：

![](vuedev.png)

## ElementUI

[官网](https://element.eleme.cn/#/zh-CN)与[官方指南](https://element.eleme.cn/#/zh-CN/component/installation)

ElementUI是由饿了么团队开源的UI框架,并于Vue完美契合，帮助你的网站快速成型。

安装可以看官方指南，或者在项目中使用命令：

```sh
npm i element-ui -S
```

如果表格ui死活出不来，安装旧版ElementUI：

```sh
cnpm uninstall element-ui
cnpm install element-ui@2.8.2
```

## FontAwesome

由于Element UI提供的字体图符较少，一般会采用其他图表库，如著名的Font Awesome 提供了675个可缩放的矢量图标，可以使用CSS所提供的所有特性对它们进行更改，包括大小、颜色、阴影或者其他任何支持的效果。

安装：

```sh
npm install font-awesome
```

在`main.js`导入：

```
import 'font-awesome/css/font-awesome.css'
```

对于4.7版本，可以使用的[图标](https://fontawesome.com.cn/v4/icons)

## Axios

在实际项目开发中，前端页面所需要的数据往往需要从服务器端获取，这必然涉及与服务器的通信。

Axios 是一个基于 promise 的网络请求库，可以用于浏览器和 node.js

Axios 在浏览器端使用XMLHttpRequests发送网络请求，并能自动完成JSON数据的转换。

![](axios.jpeg)

> Ajax是一种基于原生的XMLHttpRequest对象的技术，而Axios是一个基于Promise的HTTP客户端库。可以看作是一个更现代的Ajax

### 安装与用法

```bash
npm install axios
```

发送请求实例：

```js
// 向给定ID的用户发起请求
axios.get('/user?ID=12345')
  .then(function (response) {
    // 处理成功情况
    console.log(response);
  })
  .catch(function (error) {
    // 处理错误情况
    console.log(error);
  })
  .finally(function () {
    // 总是会执行
  });
```

```javascript
axios.post('/user', {
    firstName: 'Fred',
    lastName: 'Flintstone' // 自动转为json
  })
  .then(function (response) {
    console.log(response);
  })
  .catch(function (error) {
    console.log(error);
  });
```

### 与Vue结合

1. 在创建时请求数据
2. 传到data中
3. 模板在渲染时使用data中数据

```html
<script>
import axios from 'axios'

export default{
  data: function () {
    return{
      name_data: 'QMS',
      // name: "Zhang", deptName: "Comp. Sci.", totCred: "102", takes: null, id: "00128"
      studentsData: []
    }
  },
  created: function(){
    // 从后端获取数据，注意需要跨域
    axios.get("http://localhost:8088/student").then(response => {
      this.studentsData = response.data
    }).catch(error => {
      console.log(error)
    })
  }
}
</script>

<template>
  <h1>{{ name_data }}</h1>
  <!--循环展示studentsData-->
  <ul>
    <li v-for="student in studentsData" :key="student.id">
      name: {{ student.name }}, deptName: {{ student.deptName }}, totCred: {{ student.totCred }}, takes: {{ student.takes }}, id: {{ student.id }}
    </li>
  </ul>
</template>

<style>

</style>
```

在实际项目开发中，几乎每个组件中都会用到 axios 发起数据请求。此时会遇到如下两个问题： 每个组件中都需要导入 axios ；每次发请求都需要填写完整的请求路径。可以通过全局配置的方式解决上述问题，即在`main.js`中加入：

```js
//配置请求根路径
axios.defaults.baseURL 'http://api.com'
//将axios作为全局的自定义属性，每个组件可以在内部直接访问(Vue3)
app.config.globalProperties.Shttp axios
//将axios作为全局的自定义属性，每个组件可以在内部直接访问(ue2)
Vue.prototype.Shttp axios
```

之后，在组件代码中不需要导入 axios，`axios.get("http://localhost:8088/student")`改为`this.$http.get("/student")`

### 跨域问题

为了保证浏览器的安全，不同源的客户端脚本在没有明确授权的情况下，不能读写对方资源，称为同源策略，同源策略是浏览器安全的基石

> 同源策略（Sameoriginpolicy）是一种约定，它是浏览器最核心也最基本的安全功能，所谓同源（即指在同一个域）就是两个页面具有相同的协议（protocol），主机（host）和端口号（port）

当一个请求url的协议、域名、端口三者之间任意一个与当前页面url不同即为跨域，此时无法读取非同源网页的 Cookie，无法向非同源地址发送AJAX请求。跨域问题是前后端分离项目的常见问题。

解决：后端需要实现CORS接口，实现跨域

> CORS（Cross-Origin Resource Sharing）是由W3C制定的一种跨域资源共享技术标准，其目的就是为了解决前端的跨域请求。CORS可以在不破坏即有规则的情况下，通过后端服务器实现CORS接口，从而实现跨域通信。
>
> CORS将请求分为两类：简单请求和非简单请求，分别对跨域通信提供了支持。两类请求和接口要求这里不再赘述。

具体来说，最简单的解决方法是在springboot的controller类加上一个注解，这会允许所有跨域请求

```java
@RestController
@CrossOrigin // 允许跨域
public class StudentController {

    @Autowired
    private StudentMapper studentMapper;
    // .......
```

如果要更精细地设置跨域请求规则：

```java
@Configuration
public class CorsConfig implements WebMvcConfigurer{
    @Override
    public void addCorsMappings(CorsRegistry registry){
        registry.addMapping("/**")//允许跨域访问的路径
        .allowedorigins("*")//允许跨域访问的源
        .allowedMethods("POST","GET","PUT","OPTIONS","DELETE")//允许请求方法
        .maxAge(168000)//预检间隔时间
        ,allowedHeaders("*")//允许头部设置
        .allowCredentials(true);//是否发送cookie
    }
}
```

## VueRuter

[官方文档](https://v3.router.vuejs.org/zh/installation.html)

首先介绍SPA项目，SPA，就是单页面应用，就是目前前端开发最常见的一种，整个网站由一个html页面构成。三大框架Angular，Vue，React都是SPA。

vue-router是Vue.js官方的路由插件，它和vue.js是深度集成的，适合用于构建单页面应用。vue的单页面应用是基于路由和组件的，路由用于设定访问路径，并将路径和组件映射起来。传统的页面应用，是用一些超链接来实现页面切换和跳转的。在vue-router单页面应用中，则是路径之间的切换，也就是组件的切换。路由模块的本质 就是建立起url和页面之间的映射关系。

这里的路由就是SPA（single page application单页应用）的路径管理器。再通俗的说，vue-router就是WebApp的链接路径管理系统。**路由实际上就是可以理解为指向，就是我在页面上点击一个按钮需要跳转到对应的页面，这就是路由跳转**

vue-router 3.x 对应 vue 2

vue-router 4.x 对应 vue 3

安装

```sh
npm install vue-router@3
```

使用例子：

```html
<template>
  <div>
    <ul>
      <li>
        <router-link to="/my">My</router-link>
      </li>
      <li>
        <router-link to="/friends">Friends</router-link>
      </li>
      <li>
        <router-link to="/discover">Discover</router-link>
      </li>
    </ul>

    <!-- 声明占位标签 -->
    <router-view></router-view>
  </div>
</template>
```

同时，建议建立一个与components同级的文件夹router，放置`index.js`：

```js
import VueRouter from "vue-router";
import Vue from "vue";

import Discover from "../components/Discover.vue"; // 写好的组件
import My from "../components/My.vue";
import Friends from "../components/Friends.vue";
import TopList from "../components/TopList.vue";
import PlayList from "../components/PlayList.vue";
import Product from "../components/Product.vue";

Vue.use(VueRouter);

const router = new VueRouter({
    routes: [
        {path: "/", redirect: "/discover"},

        {
            path: "/discover", // 如果切换到这个链接
            component: Discover, // 显示这个组件
            children: [
                {path: "toplist", component: TopList},
                {path: "playlist", component: PlayList},
            ]
        },

        {
            path: "/my", 
            component: My,
            children: [
                {path: ":id", component: Product, props: true},
            ]
        },
        {path: "/friends", component: Friends},
    ]
});

export default router;
```

最后在`main.js`

```js
import Vue from 'vue'
import App from './App.vue'
import router from "./router/index.js";

Vue.config.productionTip = false

new Vue({
  render: h => h(App),
  router: router
}).$mount('#app')
```

> 完整示例项目见[网页](https://gitee.com/QMMMS/ipmsfcsv3/tree/demo/router-demo)

vue-router提供的导航守卫主要用来拦截导航，让它完成跳转或取消，类似后端拦截器。

```js
router.beforeEach((to,from,next)=>{
    if (to.path ==='/main'&&isAuthenticated){
        next('/login')
    }
    else{
        next()  // 直接放行：next()，强制其跳转到登录页面：next('/login')
    }
})
```

## Vuex

[官方文档](https://v3.vuex.vuejs.org/zh/)

对于组件化开发，大型应用状态往往跨越多个组件，传递状态十分麻烦。

Vuex 是一个专门为 Vue.js 应用程序开发的状态管理模式，它采用集中式存储管理应用的所有组件状态，并以相应的规则保证状态以一种可预测的方式发生变化。可以理解为：将多个组件共享的变量全部存储在一个对象里面，然后将这个对象放在顶层的 Vue 实例中，让其他组件可以使用，它最大的特点是响应式。

一般情况下，我们会在 Vuex 中存放一些需要在多个界面中进行共享的信息。比如用户的登录状态、用户名称、头像、地理位置信息、商品的收藏、购物车中的物品等，这些状态信息，我们可以放在统一的地方，对它进行保存和管理。

简单来说，Vuex用于管理分散在Vue各个组件中的数据。

![](vuex.png)

vuex 3 对应 vue 2

vuex 4 对应 vue 3

安装：

```sh
cnpm install vuex@3
```

建议建立一个与components同级的文件夹store，放置`index.js`：

```js
import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

const store = new Vuex.Store({
  state: {
    count: 0
  },
  mutations: {
    increment (state) {
      state.count++
    }
  }
})

export default store
```

在`main.js`做导入：

```js
import Vue from 'vue'
import App from './App.vue'
import store from './store/index.js'

Vue.config.productionTip = false

new Vue({
  render: h => h(App),
  store: store
}).$mount('#app')
```

组件：

```html
<template>
    <h1>
        Hello World!
        {{ count }}
        <button @click="add">+1</button>
    </h1>
</template>

<script>
export default {
    methods: {
        add() {
            this.$store.commit('increment')
        }
    },
    computed: {
        count() {
            return this.$store.state.count
        }
    }
}
</script>
```

> 完整示例项目见[网页](https://gitee.com/QMMMS/ipmsfcsv3/tree/demo/vuex-demo)

## mockjs

在做开发时，当后端的接口还未完成，前端为了不影响工作效率，手动模拟后端接口。使用mockjs模拟后端接口，可随机生成所需数据，拦截 Ajax 请求，可模拟对数据的增删改查。

适用于前后端分离项目，在已有接口文档的情况下，我们可以直接按照接口文档来开发，将相应的字段写好，在接口完成 之后，只需要改变url地址即可。

后续开发好之后，直接移除mockjs就行，前端项目不需要改变。

安装：

```
npm i mockjs
```

核心方法：

```js
Mock.mock( rurl?, rtype?, template|function( options ) )
```

- rurl，表示需要拦截的 URL，可以是 URL 字符串或 URL 正则 
- rtype，表示需要拦截的 Ajax 请求类型。例如 GET、POST、PUT、DELETE等。
- template，表示数据模板，可以是对象或字符串 
- function，表示用于生成响应数据的函数。

在项目中创建mock目录，新建index.js文件：

```js
//引入mockjs
import Mock from 'mockjs'
//使用mockjs模拟数据，第一个参数是要拦截的地址
Mock.mock('/product/search', {
    "ret":0,
    "data":
    {
        "mtime": "@datetime",//随机生成日期时间
        "score|1-800": 1,//随机生成1-800的数字
        "rank|1-100": 1,//随机生成1-100的数字
        "stars|1-5": 1,//随机生成1-5的数字
        "nickname": "@cname",//随机生成中文名字
        //生成图片
        "img":"@image('200x100', '#ffcc33', '#FFF', 'png', 'Fast Mock')"
    }
});

```

main.js里面引入：

```js
import Vue from 'vue'
import App from './App.vue'
import axios from 'axios';  // 使用axios创建的代码
import './mock/index.js'; // 使用mockjs添加引入，如果后端开发完成，这行代码可以删除

//axios.defaults.baseURL = 'http://localhost:8088'; // TODO 使用axios创建的代码，如果使用了mockjs，这里就不需要了
Vue.prototype.$http = axios; // 使用axios创建的代码
Vue.config.productionTip = false
new Vue({
  render: h => h(App),
}).$mount('#app')
```

之后，可以在模板中正常请求数据，mock会自动拦截

```vue
<template>
  <div>
    <img alt="Vue logo" :src="img"> <!-- 参数绑定 -->
    <ul>
      <li>{{ mock_data.mtime }}</li>
      <li>{{ mock_data.score }}</li>
      <li>{{ mock_data.rank }}</li>
      <li>{{ mock_data.stars }}</li>
      <li>{{ mock_data.nickname }}</li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'App',
  data: function() {
    return {
      img: '',
      mock_data: ''
    }
  },

  mounted: function() {
    console.log('App mounted!')
    this.$http.get('/product/search').then(
      (response) => {
        console.log(response),
        this.img = response.data.data.img
        this.mock_data = response.data.data
      }
    )
  }
}
</script>

<style>
</style>
```

> 完整示例项目见[网页](https://gitee.com/QMMMS/ipmsfcsv3/tree/demo/mock_demo)

## vue-element-admin

他人写好的一个后台管理系统，比较通用，可以直接二次开发。

[文档](https://panjiachen.github.io/vue-element-admin-site/zh/guide/)与[应用](https://panjiachen.github.io/vue-admin-template/#/dashboard)

如果启动失败，把nodejs版本改回16

[其他优秀项目](https://github.com/search?q=vue3-admin&type=repositories&s=stars&o=desc)

## JWT

### Session认证

![](session.png)

1. 用户向服务器发送用户名和密码。
2. 服务器验证通过后，在当前对话（session）里面保存相关数据，比如用户角色、登录时间等。
3. 服务器向用户返回一个 session_id，写入用户的 Cookie。
4. 用户随后的每一次请求，都会通过 Cookie，将 session_id 传回服务器。
5. 服务器收到 session_id，找到前期保存的数据，由此得知用户的身份。

客户端请求服务端，服务端会为这次请求开辟一块内存空间，这个对象就是session。但如果是服务器集群，或者是跨域的服务导向架构，就要求 session 数据共享，每台服务器都能够读取 session，针对此种问题一般有两种方案：

1. 一种解决方案是session 数据持久化，写入数据库或别的持久层。各种服务收到请求后，都向持久层请求数据。这种方案的优点是架构清晰，缺点是工程量比较大。
2. 一种方案是服务器不再保存 session 数据，所有数据都保存在客户端，每次请求都发回服务器。Token认证就是这种方案的一个代表。

### Token认证

![](tocken.png)

1. 客户端使用用户名跟密码请求登录，服务端收到请求，去验证用户名与密码
2. 验证成功后，服务端会签发一个 token 并把这个 token 发送给客户端
3. 客户端收到 token 以后，会把它存储起来，比如放在cookie 里或者localStorage 里 
4. 客户端每次向服务端请求资源的时候需要带着服务端签发的token 
5. 服务端收到请求，然后去验证客户端请求里面带着的token ，如果验证成功，就向客户端返回请求的数据

基于 token 的用户认证是一种服务端无状态的认证方式，服务端不用存放token 数据。 用解析 token 的计算时间换取 session 的存储空间，从而减轻服务器的压力，减少频繁的查询数据库

### JSON Web Token

JSON Web Token（简称 JWT）是一个token的具体实现方式，是目前最流行的跨域认证解决方案。 

JWT 的原理是，服务器认证以后，生成一个 JSON 对象，发回给用户，例如：

```json
{
    "name":"SMQ",
    "role":"admin",
    "TTL":"2099-12-31 23:59:59"
}
```

用户与服务端通信的时候，都要发回这个 JSON 对象。服务器完全只靠这个对象认定用户身份。 为了防止用户篡改数据，服务器在生成这个对象的时候，会加上签名Signature。

算出签名以后，把 Header、Payload、Signature 三个部分经过base64编码后拼成一个字符串，每个部分之间用"点"（`.`）分隔，就可以返回给用户。

![](jwt.png)

## 远程服务器配置

MySQL安装包：如果用的是CentOS7，在[下载界面](https://dev.mysql.com/downloads/mysql/)，选择RedHat以及对应版本，安装最大的tar包。

JDK安装包：在[下载界面](https://www.oracle.com/cn/java/technologies/downloads/)，选择linux版本的.tar.gz版本的安装包下载。

数据库信息：使用`mysqldump <数据库> > <导出目录> -uroot -p<密码>`导出数据库，记得使用cmd，再`source <.sql文件>`在远程导入。

都传到`/usr/server`目录下，要新建目录。

```sh
# 更新 yum
sudo yum update

# 卸载原有数据库
rpm -qa|grep mariadb
# mariadb-libs-5.5.68-1.el7.x86_64
rpm -e mariadb-libs-5.5.68-1.el7.x86_64 --nodeps

# 解压
tar xvf mysql-8.0.33-1.el7.x86_64.rpm-bundle.tar -C mysql

# 安装依赖
yum -y install libaio
yum -y install libncurses*
yum -y install perl perl-devel

# 安装mysql
rpm -ivh mysql-community-common-8.0.33-1.el7.x86_64.rpm
rpm -ivh mysql-community-client-plugins-8.0.33-1.el7.x86_64.rpm 
pm -ivh mysql-community-libs-8.0.33-1.el7.x86_64.rpm 
rpm -ivh mysql-community-client-8.0.33-1.el7.x86_64.rpm 
rpm -ivh mysql-community-icu-data-files-8.0.33-1.el7.x86_64.rpm 
rpm -ivh mysql-community-server-8.0.33-1.el7.x86_64.rpm 

# 启动mysql
systemctl start mysqld.service

# 自启动
systemctl enable mysqld.service

# 查看密码
cat /var/log/mysqld.log | grep password

# mysql登陆
mysql -uroot -p

# 设置密码，mysql中执行
set password='你的密码'

# 设置远程访问权限，mysql中执行
CREATE USER 'root'@'%' IDENTIFIED BY '你的密码';
grant all privileges on *.* to 'root'@'%';
flush privileges;

# 查看防火墙是否启动，如果启动则关闭
firewall-cmd --state
systemctl stop firewalld-service
systemctl disable firewalld-service

# 安装nginx
yum install epel-release
yum -y install nginx

# 启动nginx
systemctl start nginx

# 查看服务状态
systemctl status nginx
systemctl status mysqld

# 解压java安装包
tar -zvxf jdk-20_linux-x64_bin.tar.gz 

# 修改环境变量
vim /etc/profile
export JAVA_HOME=/usr/server/jdk-20.0.1
export PATH=${JAVA_HOME}/bin:$PATH
source /etc/profile

# 查看是否安装完成
java -version

# 部署前端
# 打包本地项目
npm run build:prod

# 把打包好的本地项目放到远程服务器，比如/usr/server/vue

# 配置nginx
cd /etc/nginx/conf.d
touch vue.conf
vim vue.conf

#############################################
server {
        listen 9528;
        server_name localhost;

        location / {
                root /usr/server/vue/dist;
                index index.html;
        }
}
#############################################

nginx -s reload

# 部署后端
# 打包后端项目，IDEA执行maven->package，打包好的项目在target目录下的jar包
nohup java -jar CollegeProjectSystem-0.0.1-SNAPSHOT.jar > lognName.log 2>&1 &

# 如果没报错，恭喜已经启动！日志文件在lognName.log
```

