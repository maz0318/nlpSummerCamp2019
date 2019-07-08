### 第一周

---

#### 实验一

* 运行命令： ``` python nnformnist.py ```
* 修改参数：
  * 修改隐藏层的维度为128 ：``` python nnformnist.py —hidden_size=128 ```
  * 修改学习率为1e-4：``` python nnformnist.py —lr=1e-4 ``` 

* 作业：
  * 自行修改参数，并记录每组参数下的正确率
  * 把2层神经网络变成3层
  * 形成报告（写到周报里）

#### 实验二

- 运行命令： ``` python cnnformnist.py ```
- 修改参数：
  - 修改第一个卷积输出通道数 ：``` python cnnformnist.py —out1_channel=10 ```
  - 修改第二个卷积输出通道数 ：``` python cnnformnist.py —out2_channel=20 ```

作业：

- 自行修改参数，并记录每组参数下的正确率
- 把2层卷积神经网络变成3层
- 形成报告（写到周报里）

#### 周末作业：

* 自行实现用cifar10数据集进行分类的卷积神经网络，网络结构，参数自己定义,记录好所用的参数对应的正确率。

  notes:记得固定好随机种子，调用nnformnist.py中的set_seed()函数