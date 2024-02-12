---
title: HuggingFace 🤗 工具集快速使用入门&中文任务示例
date: 2024-02-09 18:21:00 +0800

categories: [深度学习]
tags: [深度学习, 实验]
---


>- 视频课程:https://www.bilibili.com/video/BV1a44y1H7Jc
>- 视频课程补充篇:https://www.bilibili.com/video/BV1Cr4y1V7mF
>- 代码地址:https://github.com/lansinuote/Huggingface_Toturials
>- 代码地址2（做了一点修改，做完实验的结果）:[https://gitee.com/horizon-mind/qmmms-py-torch-practice/tree/master/Huggingface_Toturials](https://gitee.com/horizon-mind/qmmms-py-torch-practice/tree/master/Huggingface_Toturials)
{: .prompt-info }

要点：

1. install.ipynb: 需要安装的包
2. tokenizer.ipynb: 分词器使用，编码句子，批量编码，向字典中添加词和标记
3. datasets.ipynb: 下载、保存、使用数据集。排序、打乱、过滤、切分、分桶、列操作、转换类型、映射(map)内容。
4. metrics.ipynb: 评估指标和计算
5. pipeline.ipynb: 管道函数处理常见任务：情感分析、阅读理解、完形填空等等
6. 中文分类.ipynb：中文情感任务分类。只使用CPU训练模型。可以在 Google Colab 上面跑（免费但是需要代理），训练300个批次，需要 4 个小时，如果不想等可以在自己电脑上跑，大约半小时（i5-13500h）
7. 中文填空.ipynb：一个句子把第15个位置扣掉做完形填空。只使用CPU训练模型。在自己的电脑上跑（i5-13500h）大约15分钟
8. 中文句子关系推断.ipynb：判断两个句子是否是前后连贯关系。只使用CPU训练模型。在自己的电脑上跑（i5-13500h）大约1分钟
9. trainer.ipynb：HuggingFace 中 trainer 的使用方法，以及保存和使用训练好的模型参数。使用GLUE数据集（ General Language Understanding Evaluation benchmark）做分类任务。这个笔记本在自己电脑上没跑成功，在 Colab 上跑的，训练大约4分钟。
10. 中文分类_CUDA.ipynb：同样是中文分类，使用CUDA而不是 CPU 跑模型。在 Colab 上跑的，同样训练300个批次，只需要2分钟。
