###  2019njunlp夏令营 image caption group

---

#### 实验环境：

python 3.7

pytorch 1.0

### 第一阶段

### 第一周计划

|     Date      |                          Assignment                          |
| :-----------: | :----------------------------------------------------------: |
|      7.1      | 学习神经网络基础知识（ref:《机器学习》第5章 神经网络p97-p120 |
|      7.2      | 学习pytorch (ref:https://github.com/zergtant/pytorch-handbook      前两章) |
|     7.3**     | 利用pytorch框架基于mnist数据集做一个简单的分类任务（使用神经网络）(ref:https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py ) 具体任务看week1/readme.md 实验一 |
|      7.4      |            学习卷积神经网络（cs231n第15，16课时）            |
|     7.5**     | 利用pytorch框架基于mnist数据集做一个简单的分类任务（使用卷积神经网络）(ref:https://github.com/zergtant/pytorch-handbook      第三章第2节)。具体任务看week1/readme.md 实验二 |
| 7.6-7.7（周末 | 对这一周的学习做一个总结并写一个周报，内容包括学到了什么，还有哪些部分不太理解以及实验的记录。 |

注：带**是有实验要完成的

#### 第一周实验

* 用神经网络对mnist数据集进行分类
* 用卷积神经网络对mnist数据集进行分类



### 第二周计划

| Date            | Assignment                                                   |
| --------------- | ------------------------------------------------------------ |
| 7.8             | 学习循环神经网络RNN.(ref:https://github.com/zergtant/pytorch-handbook/blob/master/chapter2/2.5-rnn.ipynb ,视频:https://study.163.com/course/introduction.htm?courseId=1004697005#/courseDetail?tab=1 ) |
| 7.9             | 了解命名实体识别任务（NER)(ref:https://blog.csdn.net/u014033218/article/details/89304699 ) |
| 7.10**          | 用LSTM实现NER(week2/readme)                                  |
| 7.11**          | 用bi-LSTM+CRF实现NER(week2/readme)                           |
| 7.12**          | 对模型进行调整，调参，优化结果(week2/readme)                 |
| 7.13-7.14(周末) | 周报                                                         |

#### 第二周实验

* 用LSTM实现NER

* 用bi-LSTM+CRF实现NER

  

### 第三周计划

由于大家对pytorch不是很熟悉，所以给大家推荐一个pytorch教程，本周自行去学习，同时阅读一篇caption的论文。

教程：https://mlelarge.github.io/dataflowr-web/cea_edf_inria.html

dataflowr:https://mlelarge.github.io/dataflowr-web/

github:https://github.com/mlelarge/dataflowr

论文:https://arxiv.org/abs/1411.4555



### 第二阶段

### 第四周计划

本周，我们复现上周看的论文show and tell

| Date | Assignment                                    |
| ---- | --------------------------------------------- |
| 7.22 | 对数据集进行预处理，放在process.py里          |
| 7.23 | 对coco数据集写一个dataset放在data_loader.py里 |
| 7.24 | 复现模型，放在model.py里                      |
| 7.25 | 写训练主程序，放在train_nic.py里,并开始训练   |
| 7.26 | 写好beam search，放在model.py里               |
| 7.27 | 查看训练效果，写好报告                        |

