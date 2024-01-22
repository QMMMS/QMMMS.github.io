---
title: How to Break the Limitations of Google Search？
date: 2024-01-22 11:21:00 +0800

img_path: "/assets/img/posts/2024-01-22-2024-01-22-How to Break the Limitations of Google Search"
categories: [计算机网络]
tags: []
---

## Review

As we all know, in one of the greatest countries of the world, almost all of the software engineers there must learn proxy and computer network knowledge well to get access to good enough learning materials such as code or documents on foreign websites.

How to get a VPN to get access to websites like Google is very basic so are not discussed here again. However, to better understand what we will do next, we better review the theory behind VPN. This is a [link](https://qmmms.github.io/posts/%E8%99%9A%E6%8B%9F%E4%B8%93%E7%94%A8%E7%BD%91%E7%BB%9C%E5%8E%9F%E7%90%86/) to my previous post about VPN.

And we can have an insight about the whole picture. If I use the noraml network provided by my ISP, my IP looks like this:

![](1.png)
_IP in Normal Network_

What if I use a VPN? My IP looks like this:

![](2.png)
_IP in VPN_

Well, I must say that this VPN is not good enough. We can see the host name begins with `vps`, and it is like I'm announcing to the world, especially to the Great Fire Wall, that I'm using a VPN.

Whatever, it is free, don't ask for too much.

## Problem

But, there's a problem I can't tolerate. Though I can use Google Search, the results are filtered. I'v found a setting called `Safe Search` in Google Search Settings, and I can't turn it off.

> You can't change your SafeSearch setting right now because someone else, like a parent or administrator, controls settings on the network, browser, or device you’re using.

## DuckDuckGo

The easiest way to solve this problem is to use another search engine. DuckDuckGo can be a good choice.

> The DuckDuckGo app includes our Web and App Tracking Protection, Smarter Encryption, Private Search, Email Protection, and more.

This is a [link](https://duckduckgo.com/) to their website. If you are in specific areas, you will need a vpn to get access to it.

## Tor

The problem seems to be solved, but I want to do futher. In fact, I heard of DuckDuckGo form another famous tool, Tor. Tor uses DuckDuckGo as its default search engine.

Here are steps to use Tor in specific areas:

1. Download Tor Browser from [here](https://www.torproject.org/download/).
2. If you are in specific areas, you will have trouble connecting to the Tor network.  ![](3.png)
3. You can use a bridge to solve this problem. click `Settings` and choose the `Connection` tab, then `select a buildin bridge`, `meek-azure` works in my case.
4. The first time you use Tor, you will have to wait for a long time to connect to the Tor network. But after that, your connection will be faster.
5. Note that the whole process you will **not** need a VPN, if you have reviewed the theory behind it, you will know why.
6. You've done it!

![](4.png)

And if you see the path of your connection, you will find your message is sent to several nodes in different countries, and then to the destination. And if you use a IP address checker, every time you connect to the Tor network, your IP address will be different.

![](5.png)
