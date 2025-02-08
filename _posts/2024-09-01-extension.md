---
title: Big Picture of Flarum Extensions
date: 2024-09-01 12:21:00 +0800

media_subpath: "/assets/img/posts/2024-09-01-extension"
categories: [经验与总结]
tags: [经验, LLM]
---

> 为 Flarum 编写插件所需要的基础知识与实例
{: .prompt-info }

## 支撑组件

**PHP**是在服务器端运行的脚本语言，与HTML紧密结合。开发人员可以在HTML中嵌入PHP代码，并在服务器上进行解析和执行。这种结合使得开发人员能够轻松地生成动态的Web页面，根据用户的请求动态生成内容。与C、C++语言有着相似的语法结构，与许多数据库管理系统（DBMS）兼容，如MySQL、Oracle、SQLite等。多线程支持。

**Composer**是一个非常流行的PHP包依赖管理工具,已经取代PEAR包管理器。对于使用者来说Composer非常的简单,通过简单的一条命令将需要的代码包下载到vendor目录下,然后开发者就可以引入包并使用。其中的关键在于你项目定义的composer.json,可以定义项目需要依赖的包。

> 每个 Flarum 扩展也是一个 Composer 包。这意味着 Flarum 安装需要某个扩展时，可以使用 Composer 将其引入并保持最新。
>
> 例如，在composer.json中定义`"name": "acme/flarum-hello-world"`之后，可以通过`composer require acme/flarum-hello-world *@dev`命令导入项目

**Laravel**（读音：拉拉维尔）是一个基于PHP的开源Web应用程序框架，它遵循MVC（模型-视图-控制器）设计模式。提供了丰富的功能特性，Flarum 使用到了 Laravel 的数据库组件、事件系统、前端 Blade 模板，以及使用Laravel 的服务容器（或 IoC 容器）进行依赖项注入。

> 其他流行的Web框架：如Django（Python）、Ruby on Rails（Ruby）、Spring Boot（Java）和Express（Node.js）
>
> Laravel 的数据库组件select实例：
>
> ```php
> class UserController extends Controller{
>     public function index(): View {
>         $users = DB::select('select * from users where active = ?', [1]);
>         return view('user.index', ['users' => $users]);
>     }
> }
> ```

**Mithril.js** 是一个现代的客户端JavaScript框架，专为构建单页应用程序（SPA）而设计。它以其小巧的体积（压缩后仅9.17 KB），高效的性能以及内置的路由和XHR工具而受到赞誉。

## Flarum设计思想

> [参考1](https://docs.flarum.org/extend)、[参考2](https://docs.flarum.org/extend/start)

Flarum 的构成有三层： 

- 第一层，后端。 后端用 面向对象的 PHP 语言编写，并通过 Composer 使用了大量的 Laravel 组件和其他资源包。依赖项注入的概念在整个后端中都有使用。 
- 第二层，后端开放的一个 公共 API，允许前端客户端与论坛数据进行交互。 该接口根据 JSON:API 规范 构建。 
- 第三层，Flarum 的前端是**单页 JavaScript 应用程序**， 由一个简单的类 React 框架 Mithril.js 构建。有两个独立的前端应用程序：
  - `forum` ，论坛的公共部分，用户可以在其中创建讨论和帖子。
  - `admin` ，论坛的私人部分，作为论坛管理员，您可以在其中配置 Flarum 安装。

**Flarum 的核心**并不旨在包含所有功能。相反，它是一个脚手架或框架，为构建扩展提供了可靠的基础。它只包含论坛所必需的基本的、不带偏见的功能：讨论、帖子、用户、组和通知。

**捆绑扩展**是与 Flarum 一起打包并默认启用的功能。它们是与其他扩展一样的扩展，可以被禁用和卸载。虽然它们的范围并不旨在解决所有用例，但其想法是使它们足够通用和可配置，以满足大多数人的需求。

**第三方扩展**是由其他人开发的功能，不受 Flarum 团队的正式支持。它们可以被构建并用于解决更具体的用例。

## 环境准备

> Windows环境

1. 安装[php](https://windows.php.net/download#php-8.3)，配置环境变量
2. 安装[composer](https://getcomposer.org/download/)
3. 安装nodejs和npm
4. 使用[Flarum CLI](https://discuss.flarum.org/d/28427-flarum-cli-v10)初始化开发 Flarum 扩展的环境
   - 安装新包：`npm install -g @flarum/cli`
   - 初始化环境：`flarum-cli init`
5. 使用 phpstrom IDE 进行代码编辑

## 扩展器

为了扩展 Flarum，Flarum开发者使用一个称为**扩展器**的概念。扩展程序是*声明性*对象，它们以简单的方式描述试图实现的目标（例如向论坛添加新路线，或在创建新讨论时执行一些代码）

```php
// 向前端注册 js 和 css
(new Extend\Frontend('forum'))
    ->js(__DIR__.'/forum-scripts.js')
    ->css(__DIR__.'/forum-styles.css')
```

一个最简单的例子是使用Flarum 安装根目录中的`extend.php`，它应该返回扩展器对象的数组，下面这个例子中，它通过找到dom对象添加了一行js代码：

```php
<?php

use Flarum\Extend;
use Flarum\Frontend\Document;

return [
    (new Extend\Frontend('forum'))
        ->content(function (Document $document) {
            $document->head[] = '<script>alert("Hello, world!")</script>';
        })
];
```

## 前端工作流

在开发环境中，一个典型的插件的前端工程结构如下：

```
js
├── dist (compiled js is placed here)
├── src
│   ├── admin
│   └── forum
├── admin.js
├── forum.js
├── package.json
├── tsconfig.json
└── webpack.config.js
```

admin.js 和 forum.js 文件包含实际前端 JS 的根目录。虽然可以将整个扩展放在这里，但这不是最佳实践。Flarum 开发者建议将实际的源代码放在`src`中，并让这些文件仅导出`src`的内容。例如：

```typescript
// forum.js
export * from './src/common';
export * from './src/forum';
```

如果遵循上面的建议，我们希望`src`有 2 个子文件夹：一个用于`admin`前端代码，一个用于`forum`前端代码。`common`子文件夹放置两个前端之间共享的组件、模型、实用程序或其他代码。`admin`和`forum`的结构是相同的，仅在此处显示`forum`的结构：

```
src/forum/
├── components/
|-- models/
├── utils/
└── index.js
```

这里最重要的文件是`index.js` ：其他所有文件只是将类和函数提取到自己的文件中。让我们看一下典型的`index.js`文件结构：

```javascript
import { extend, override } from 'flarum/common/extend';

// 在core启动后，这段代码会callback回调
app.initializers.add('acme-flarum-hello-world', function(app) {
  // 你自己的扩展代码
  console.log("EXTENSION NAME is working!");
});
```

在代码编写完成之后，使用命令将浏览器就绪的 JavaScript 代码编译到`js/dist/forum.js`文件中：

```bash
npm install
npm run dev
```

最后，为了将扩展的 JavaScript 加载到前端，需要告诉 Flarum 在哪里可以找到它。可以使用`Frontend`扩展器的`js`方法来做到这一点。将其添加到扩展的`extend.php`文件中：

```javascript
<?php
use Flarum\Extend;
return [
    (new Extend\Frontend('forum'))
        ->js(__DIR__.'/js/dist/forum.js')
];
```

## 前端代码原理

Flarum 的界面是使用名为[Mithril.js](https://mithril.js.org/)的 JavaScript 框架构建的。关键在于 Flarum 生成虚拟 DOM 元素， Mithril 采用这些虚拟 DOM 元素，并以最有效的方式将它们转换为真实的 HTML。

因为界面是用 JavaScript 构建的，所以很容易挂接并进行更改。需要做的就是为要更改的界面部分找到正确的 extender ，然后将自己的虚拟 DOM 添加到其中。界面的大多数可变部分实际上只是*项目列表*。例如：

- 每个帖子上显示的控件（回复、点赞、编辑、删除）
- 索引侧边栏导航项（所有讨论、关注、标签）
- 标题中的项目（搜索、通知、用户菜单）

例如，下面的这个`index.js`将 Google 链接添加到标题：

```javascript
import { extend } from 'flarum/common/extend';
import HeaderPrimary from 'flarum/forum/components/HeaderPrimary';

extend(HeaderPrimary.prototype, 'items', function(items) {
  items.add('google', <a href="https://google.com">Google</a>);
});
```

在上面的示例中，我们使用`extend`实用函数（如下所述）将 HTML 添加到 `HeaderPrimary.prototype.items()` 。我们继续分析这段小程序，从component开始。当然，如果不想看，前端代码原理这一小节的剩余部分可以跳过

## 前端与Component

Flarum 的界面是由许多嵌套的**组件**（Component）组成的。组件有点像 HTML 元素，因为它们封装了内容和行为。例如，看看组成讨论页面的组件的简化树：

```
DiscussionPage
├── DiscussionList (the side pane)
│   ├── DiscussionListItem
│   └── DiscussionListItem
├── DiscussionHero (the title)
├── PostStream
│   ├── Post
│   └── Post
├── SplitDropdown (the reply button)
└── PostStreamScrubber
```

Flarum 将组件包装在`flarum/common/Component`类中，要使用 Flarum 组件，只需在自定义组件类中扩展`flarum/common/Component`即可。自定义计数器组件类可能如下所示：

```javascript
import Component from 'flarum/common/Component';

class Counter extends Component {
  oninit(vnode) {
    super.oninit(vnode);
    this.count = 0;
  }

  view() {
    return (
      <div>
        Count: {this.count}
        <button onclick={e => this.count++}>
          {this.attrs.buttonLabel}
        </button>
      </div>
    );
  }

  oncreate(vnode) {
    super.oncreate(vnode);
    $element = this.$();
    $button = this.$('button');
  }
}

m.mount(document.body, <MyComponent buttonLabel="Increment" />);
```

> 注意，扩展`Component`组件类在使用生命周期方法（ `oninit` 、 `oncreate` 、 `onbeforeupdate` 、 `onupdate` 、 `onbeforeremove`和`onremove` ）时必须调用`super`

在编写好自己的Component后就可以在extend函数中使用了。

## 前端与extend

Flarum 包含`extend`和`override`实用函数。 `extend`允许我们添加代码以在方法完成后运行。 `override`允许我们用新方法替换方法，同时保持旧方法可用作回调。两者都是带有 3 个参数的函数：

- 类（或其他一些可扩展对象）的原型
- 该类中方法的字符串名称
- 执行修改的回调
  - 对于`extend` ，回调接收原始方法的输出，以及传递给原始方法的任何参数
  - 对于`override` ，回调接收可调用对象（可用于调用原始方法）以及传递给原始方法的任何参数

现在让我们重新回顾一下原来的“将 Google 链接添加到标题”示例进行演示

```javascript
import { extend, override } from 'flarum/common/extend';
import HeaderPrimary from 'flarum/forum/components/HeaderPrimary';
import ItemList from 'flarum/common/utils/ItemList';
import CustomComponentClass from './components/CustomComponentClass';

// 我们在返回的 ItemList 中添加了一个项目
// 使用了上面提到的自定义 Component
// 还指定了优先级作为第三个参数，这将用于对这些项目进行排序
// 请注意，不需要返回任何东西
extend(HeaderPrimary.prototype, 'items', function(items) {
  items.add(
    'google',
    <CustomComponentClass>
      <a href="https://google.com">Google</a>
    </CustomComponentClass>,
    5
  );
});

// 根据条件使用方法的原始输出，或者创建我们自己的 ItemList
// 然后将一个项目添加到其中
// 请注意，我们必须返回我们自定义的输出
override(HeaderPrimary.prototype, 'items', function(original) {
  let items;

  if (someArbitraryCondition) {
    items = original();
  } else {
    items = new ItemList();
  }

  items.add('google', <a href="https://google.com">Google</a>);
  return items;
});
```

## 典型 API 请求的生命周期

![](api_flowchart.png)

1. HTTP 请求被发送到 Flarum 的 API。通常，这将来自 Flarum 前端，但外部程序也可以与 API 交互。 Flarum 的 API 大多遵循[JSON:API](https://jsonapi.org/)规范，因此相应地，请求也应遵循[该规范](https://jsonapi.org/format/#fetching)。
2. 该请求通过[中间件](https://docs.flarum.org/extend/middleware)（middleware）运行，并路由到适当的控制器。
3. 通过[`ApiController`扩展器](https://docs.flarum.org/extend/api#extending-api-controllers)对控制器进行扩展所做的任何修改都会被应用。这可能需要更改排序、添加包含、更改序列化器等。
4. 调用控制器的`$this->data()`方法，生成一些应返回给客户端的原始数据。通常，这些数据将采用 Laravel Eloquent 模型集合或实例的形式，这些数据是从数据库中检索的。话虽这么说，数据可以是任何数据，只要控制器的串行器（controller's serializer）可以处理它即可。每个控制器负责实现自己的`data`方法。请注意，对于`PATCH` 、 `POST`和`DELETE`请求， `data`将执行相关操作，并返回修改后的模型实例。
5. 该数据通过扩展通过[`ApiController`](https://docs.flarum.org/extend/api#extending-api-controllers)扩展程序注册的任何预序列化回调运行。
6. 数据通过[序列化器](https://docs.flarum.org/extend/api#serializers)传递，序列化器将其从后端数据库友好的格式转换为前端期望的 JSON:API 格式。它还附加任何相关对象，这些对象通过自己的序列化器运行。
7. 序列化数据作为 JSON 响应返回到前端。
8. 如果请求是通过 Flarum 前端的`Store`发起的，则返回的数据（包括任何相关对象）将作为[前端模型](https://docs.flarum.org/extend/api#frontend-models)存储在前端存储中。

## 后端与Controllers

什么是控制器（Controllers）？在这个项目中，控制器的最重要的`handle`方法是当有人访问 route 路由（或通过表单提交向其发送数据）时运行的代码。一般来说，控制器的实现遵循以下模式：

1. 从 Request 对象检索信息（GET 参数、POST 数据、当前用户等）。
2. 利用该信息做一些事情。例如，如果我们的Controller处理创建帖子的路由，我们将希望将新的帖子对象保存到数据库中
3. 返回响应。大多数路由将返回 HTML 网页或 JSON api 响应。

例如，`extend.php`可以包含如下代码，在访问`/hello-world`路由时（前端也需要配置路由）由`HelloWorldController`响应：

```php
return [
    (new Extend\Routes('forum'))
        ->get('/hello-world', 'acme.hello-world', HelloWorldController::class)
];
```

`HelloWorldController`代码如下：

```php
class HelloWorldController implements RequestHandlerInterface {
    public function handle(Request $request): Response {
        return new HtmlResponse('<h1>Hello, world!</h1>');
    }
}
```

如果要使用数据库，Flarum 中的 **Migrations** 允许修改数据库，例如添加新表、定义新关系、向表添加新列或进行其他数据库结构更改。**Models** 提供了一个方便的、基于代码的 API，用于创建、读取、更新和删除数据。在后端，它们由 PHP 类表示，用于与 MySQL 数据库交互。在前端，它们由 JS 类表示，并用于与[JSON:API](https://docs.flarum.org/extend/api)交互。

流程就是在 **controller** 中做数据库相关操作，然后通过 Serializers 序列化将其转换为 JSON:API 格式，以便前端可以使用它。Flarum 的前端包含一个本地数据`store` ，它提供了与 JSON:API 交互的接口。加载资源后，它们将被缓存在存储中，以便再次访问它们。

## 后端事件

通常，扩展会希望对 Flarum 其他地方发生的某些事件做出反应。例如，我们可能希望在发布新讨论时增加计数器，在用户首次登录时发送欢迎电子邮件，或者在将讨论保存到数据库之前向讨论添加标签。这些事件称为**领域事件**，并通过[Laravel 的事件系统](https://laravel.com/docs/8.x/events)在整个框架中广播。

例如，`extend.php`可以写入如下代码：

```php
return [
    (new Extend\Event)
        ->listen(Deleted::class, PostDeletedListener::class)
];
```

在这里，我们可以使用侦听器类代替回调函数。这允许通过构造函数参数将[依赖项注入](https://laravel.com/docs/8.x/container)到侦听器类中。

```php
class PostDeletedListener {
    protected $translator;
    public function __construct(TranslatorInterface $translator) {
        $this->translator = $translator;
    }
    public function handle(Deleted $event) {
        // 扩展代码
    }
}
```

还可以通过**事件订阅者** subscriber 同时监听多个事件。这对于对常见功能进行分组很有用；例如，如果您想更新帖子更改的一些元数据：

```php
return [
    (new Extend\Event)
        ->subscribe(PostEventSubscriber::class),
];
```

```php
class PostEventSubscriber {
    protected $translator;
    public function __construct(TranslatorInterface $translator){
        $this->translator = $translator;
    }
    public function subscribe($events) {
        $events->listen(Deleted::class, [$this, 'handleDeleted']);
        $events->listen(Saving::class, [$this, 'handleSaving']);
    }
    public function handleDeleted(Deleted $event) {/*...*/}  
    public function handleSaving(Saving $event) {/*...*/}
}
```

调度事件非常简单。所需要做的就是注入 `Illuminate\Contracts\Events\Dispatcher` 进入新的类，然后调用它的`dispatch`方法。例如：

```php
class SomeClass {
    protected $events;
    public function __construct(Dispatcher $events) {
        $this->events = $events;
    }
    public function someMethod() {
        // Logic
        $this->events->dispatch(
            new Deleted($somePost, $someActor)
        );
        // More Logic
    }
}
```

## Service Provider

正如本文档中所述，Flarum 使用[Laravel 的服务容器](https://laravel.com/docs/8.x/container)（或 IoC 容器）进行依赖项注入。Service Provider 允许对 Flarum 后端进行低级配置和修改。

要了解 Service Provider，首先需要了解 Flarum 的启动顺序：

1. 容器和应用程序已初始化，并且注册了基本绑定（配置、环境、记录器）
2. 运行所有核心 Service Provider 的`register`方法
3. 运行所有启用的扩展所使用的所有扩展器的`extend`方法
4. Flarum 站点本地的`extend.php`中使用的所有扩展器的`extend`方法都会运行
5. 运行所有核心 Service Provider 的`boot`方法

自定义的 Service Provider 应该扩展 `Flarum\Foundation\AbstractServiceProvider` ，并且可以有一个`boot`和一个`register`方法。例如：

```php
// extend.php
return [
    (new Extend\ServiceProvider())
        ->register(CustomServiceProvider::class),
];
```

```php
class CustomServiceProvider extends AbstractServiceProvider {
    public function register() {
        // 自定义逻辑代码，例如:
        $this->container->resolving(SomeClass::class, function ($container) {
            return new SomeClass($container->make('some.binding'));
        });
    }
    public function boot(Container $container) {
        // 自定义逻辑代码
    }
}
```

`register`方法将在上面的步骤(3)中运行， `boot`方法将在上面的步骤(5)中运行。在`register`方法中，可以通过`$this->container`获取容器。在`boot`方法中，容器（或任何其他参数）应通过类型提示的方法参数注入。

## 实例：OpenAI Agent

> [参考](https://github.com/blomstra/flarum-ext-support-ai)

**核心需求**：编写插件，在论坛用户发帖/评价后由人工智能助手 Agent 自动回复一个楼层

**需求细节**：

- 历史记忆：类似大模型对话系统，在生成回复时需要有之前所有楼层的信息
- 更细节的控制：全部回复、只回复帖子的第一个楼层、只在允许的板块中回复、回复用户@
- 内容审核：作为一个 content moderator，对于不符合相关法律法规的回复，不自动回复，而是列出违反了规则的原因，并且触发事件以供后续处理
- 导出微调数据：为了以后微调大模型，需要创建自定义控制台命令（Console command），导出最适合大模型微调的论坛交流实例数据

**实现方法**：

在`extend.php`注册事件侦听器或者事件订阅者，监听Posted事件。

```php
(new Flarum\Event)->subscribe(Listen\ReplyToPosts::class)
```

对于需求细节中 *更细节的控制* ，使用权限的思想来判断是否需要由 Agent 自动回复，这里的权限的本质是字符串枚举类，对于使用者，通过修改前端的方式可以在论坛管理页面进行权限设置：

```php
enum Permission: string {
    case REPLY_TO_MENTIONS = 'discussion.supportAiRespondToMentions';
    case REPLY_TO_OP = 'discussion.supportAiRespondToOp';
    case REPLY_TO_REPLIES = 'discussion.supportAiRespondToReplies';
}
```

在满足权限后，新建 ReplyJob 回复任务并放入类似任务队列中的 queue 中，其中ReplyJob 回复任务的核心代码是 handle 方法，使用 Agent 回复目标帖子：

```php
$this->queue->push(new ReplyJob($event->post));
```

```php
public function handle(Agent $agent): void {
    $agent->repliesTo($this->post);
}
```

对于需求细节中 *历史记忆* ，我们遍历 post 中的 `discussion->comments()`，拿出每一个 CommentPost，转化为 Message（包含了智能体的角色和内容）放到一个 Collection 中。

```php
$post->discussion->comments()  // 对于每一条评论
    ->whereVisibleTo(new Guest)  // 如果对访客可见
    ->whereKeyNot($post)
    ->when($post->discussion->firstPost, fn ($query, $firstPost) => $query->whereKeyNot($firstPost))
    ->each(fn (CommentPost $comment) => $messages->push(self::buildFromPost($comment)));  // 将评论转换为 Message
```

对于需求细节中 *内容审核* ，我们定制在当前的 instruction，`$moderator` 和 `$persona` 来自论坛中的具体设定，当违反时，大模型会返回以`FLAG: `作为开头的具体原因：

```
You are tasked with reviewing posts made by people on a community. Your task is twofold, on one side you will review
the text based on moderation instructions and on the other hand you will fulfill the role of an assistant.
=====
here follow the instructions as a content moderator, in case you consider the content of the user to breach these instructions reply with 'FLAG: ' and the reason you consider this to breach the instructions: $moderator;
=====
Here follow the instructions as an assistant: $persona
```

然后，让大模型生成response：

```php
$response = $this->client->chat()->create([
    'model' => 'gpt-3.5-turbo',
    'messages' => $messages,
    'user' => "user-$post->user_id"
]);
```

我们可以检查生成的response，如果以`FLAG: `作为开头，触发Flagging、Created事件：

```php
static::$events->dispatch(new Flagging($this));
static::$events->dispatch(new Created($flag, static::$agent->user, []));
```

如果符合社区规则，触发 Replying 回复事件，并回复帖子：

```php
CommentPost::reply(
    discussionId: $post->discussion_id,
    content: $reply(),  // 调用 __invoke 方法中 static::$events->dispatch(new Replying($this, $message));
    userId: $this->user->id,
    ipAddress: '127.0.0.1',
    actor: $this->user
)->save();
```

至此，主要功能的实现方法介绍完毕，下面聊聊代码中的其他细节。

对于需求细节中 *导出微调数据*，可以通过在 `extend.php` 注册 [Console Commands](https://docs.flarum.org/extend/console/#registering-console-commands) 来实现：

```php
(new Flarum\Console)->command(Console\TrainAgentCommand::class)
```

`TrainAgentCommand` 类继承自 `Command` 类，实现了 `handle` 函数，内部重要代码包括查询数据库：

```php
Discussion::query()  // 开始查询
    ->with('firstPost', 'bestAnswerPost')
    // 预加载（Eager Loading） firstPost 和 bestAnswerPost 关系，优化数据库查询
    ->whereNotNull('discussions.best_answer_post_id')  // 只查询有最佳答案的讨论
    ->latest('discussions.created_at')  // 按创建时间倒序排序
    ->limit(50)  // 限制50条
    ->get()  // 获取结果
    ->map(function (Discussion $discussion) {  // 遍历结果，注意最后返回Collection
        return [
            $discussion->id,
            $discussion->firstPost->content,
            $discussion->bestAnswerPost->content
        ];
    });
```

拿到论坛交流实例数据后就简单了，可以直接输出，或者转为嵌入向量，后面的代码不再赘述

```php
// 使用 openai 的 embeddings 方法创建嵌入向量
$embedding = $client->embeddings()->create([
    'model' => 'text-embedding-ada-002',
    'input' => $data
    ->prepend(['id', 'question', 'answer'])
    ->toJson()
]);
```

此外值得一提的是，可以使用 ServiceProvider 来做依赖注入，例如在 `extend.php` 中：

```php
(new Flarum\ServiceProvider)
    ->register(BindingsProvider::class)
    ->register(ClientProvider::class), 
```

复习一下，Flarum 的启动流程中，会自动调用 `ServiceProvider` 的 `boot` 方法对 Flarum 后端进行低级配置和修改，例如：

```php
// 将匿名函数注册为容器中的单例服务
// 注册单例服务：将 Client 类的实例注册为容器中的单例服务。
// 延迟实例化：只有在第一次请求 Client 实例时，箭头函数才会被调用，创建并返回 Client 实例。
// 依赖注入：通过容器管理依赖关系，确保 Client 实例在整个应用生命周期中是唯一的。
$this->container->singleton(Client::class, fn () => OpenAI::client($apiKey, $organisation));

/** @var Client $client */
$client = $container->make(Client::class);  // 从容器中获取 OpenAI Client实例并注入
```

关于 OpenAI 需要的 api key，同样可以通过修改前端的方式，在论坛管理页面进行设置，创建的适合把论坛设置拿过来用就可以：

```php
$this->container->singleton(Agent::class, fn () => $this->getAgent($settings, $extensions));

protected function getAgent(SettingsRepositoryInterface $settings, ExtensionManager $extensions): Agent{
    
    // 判断容器中是否已经有了 Client 类
    $client = $this->container->has(Client::class)
        ? $this->container->make(Client::class)
        : null;
    
    $agent = new Agent(
        user: $user,
        persona: $settings->get('blomstra-support-ai.persona'),
        moderatingBehaviour: $settings->get('blomstra-support-ai.how-to-moderate'),
        client: $client,
        model: $settings->get('blomstra-support-ai.model')
    );
    
    return $agent;
}
```

