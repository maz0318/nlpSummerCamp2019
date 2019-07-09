### 第二周

---

#### 实验一

用LSTM实现命名实体识别任务

数据集下载：https://pan.baidu.com/s/1BU0XS-I5qZIA7Y9trGxc8w 提取码: gnqt

数据集使用说明:https://github.com/zjy-ucas/ChineseNER

* Tips:

  * LSTM可用pytorch中的torch.nn.LSTM
  * 最后的评判标准要计算precision，recall和f1-score

  * 这个数据集Pytorch里事先没有集成，所以要自己写dataset。