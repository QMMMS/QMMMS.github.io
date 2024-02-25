---
title: Clash For Linux
date: 2024-02-25 11:21:00 +0800

categories: [计算机网络]
---

深度学习需要数据，国外论文的很多数据是保存在谷歌 drive 上面的，需要让Linux服务器具备上网功能。

> - 推荐上网工具：https://cdn.runba.cyou/user
> - 参考：https://cdn.runba.cyou/doc/#/linux/clash

## 创建并进入程序目录

```bash
mkdir  ~/.config/
mkdir  ~/.config/mihomo/
cd     ~/.config/mihomo/
```

## 下载clash

> 参考中提供的是 linux-amd64 版本，如果你启动不了，可能是不适合你的系统，你可以从[官网下载其它的版本](https://github.com/MetaCubeX/mihomo/releases)。 根据Linux版本选择相应的下载。

```bash
curl -# -O https://cdn.runba.cyou/ssr-download/clash-linux.tar.gz
tar xvf clash-linux.tar.gz  # 解压
chmod +x clash-linux  # 授权可执行权限
```

## 下载 clash 配置文件(更新订阅更新节点)

用wget下载clash配置文件（**重复执行就是更新订阅更新节点**），替换默认的配置文件。当然，你也可以用浏览器打开订阅链接，下载后拷贝或移动到~/.config/mihomo/目录替换覆盖config.yaml文件。

下载配置文件：（如果下载失败,试试将前缀更换为 `https://cdn.runba.cyou`）：

```bash
wget -U "Mozilla/6.0" -O ~/.config/mihomo/config.yaml 订阅链接
```

然后，启动clash【切记：不要加 sudo】

```
./clash-linux
```

## Linux命令行设置代理

clash启动已占用的终端窗口无法再输入命令，请新开一个终端窗口执行下列命令。

在Linux命令行中设置代理，可以通过设置环境变量http_proxy和https_proxy来实现（下列命令只对当前终端窗口有效，如果希望永久性的设置代理，可以将以上命令添加到.bashrc文件中）：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

输入 `echo $http_proxy` 和 `echo $https_proxy` 命令，然后回车查看，以确保代理已经正确设置。

为了测试，尝试访问 Google（ping 不支持代理，命令行测试外网网址请使用 curl 测试）

```bash
curl www.google.com
```

如果需要取消代理，可以使用以下命令：

```bash
unset http_proxy
unset https_proxy
```

## Clash Web 管理

**下载代码：**

```bash
mkdir /etc/clash/  && cd /etc/clash/
curl -# -O https://cdn.runba.cyou/ssr-download/clash-dashboard.tar.gz
tar zxvf clash-dashboard.tar.gz
```

**修改配置：**不需要修改 Clash Dashboard 的文件，需要修改的是 Clash 的配置文件。一般情况下是没有配置 external-ui 和 secret 这两个配置，编辑配置文件进行查看，如果没有就加入配置，如果有的话查看 external-ui 的路径是否正确；还需要将 external-controller 的地址修改为：127.0.0.1:9090 如果你不是从本机访问，需要从其它机器访问这个Clash Dashboard ,则改为：0.0.0.0:9090

```bash
cd ~/.config/mihomo/ 
vim config.yaml 

# 在配置文件中修改或增加以下内容；
external-controller: 127.0.0.1:9090 # 如果你不是从本机访问，需要从其它机器访问这个Clash Dashboard ,则改为：0.0.0.0:9090
external-ui: /etc/clash/clash-dashboard # clash-dashboard的路径；
secret: 'PaaRwW3B1Kj9' # PaaRwW3B1Kj9 是登录web管理界面的密码，请自行设置你自己的,不要照抄教程中的密码；
# 重启clash
```

**访问测试：**Clash Dashboard 的本机访问地址是：`127.0.0.1:9090/ui` , 注意:本机访问浏览器地址栏和页面中的host字段都是 `127.0.0.1` ,如果是从其它机器访问,则需要将 两处的 `127.0.0.1` 都改为Clash机器的IP。

如果是使用 AutoDL 的深度学习服务器，IP 不对外开放，怎么办？可以在AutoDL控制台网页点击`自定义服务`，下载 SSH 隧道工具，填上 SSH 指令、密码和代理端口。就可以将 AutoDL 的深度学习服务器指定端口映射到自己的电脑上，自然，在Clash Dashboard 页面中的host字段就填的是`127.0.0.1`。

## 下载 Google Drive 文件

> 参考：https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive

```bash
pip install gdown  # 下载gdown工具
```

下载命令格式：

```bash
gdown https://drive.google.com/uc?id=<file_id>  # for files
gdown <file_id>  # alternative format
gdown --folder https://drive.google.com/drive/folders/<file_id>  # for folders
gdown --folder --id <file_id>  # this format works for folders too
```

`file_id` 应该看上去像 `0Bz8a_Dbh9QhbNU3SGlFaDg`. You can find this ID by right-clicking on the file of interest, and selecting *Get link*. 

例子：

```bash
gdown --fuzzy https://drive.google.com/file/d/1DS1nof3lhq5QiWRrvOTiN436_ca4EN2Y/view?usp=sharing
```

