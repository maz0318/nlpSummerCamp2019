### 第四周

---

coco数据集已经放在服务器上，在/home/user_data55/maz/cocodata/里

#### process.py

对coco数据集进行预处理

coco提供了一个共我们可用的api    --pycocotools

可使用pip安装 ``` pip install pycocotools ```

处理的时候要用设计3个特殊字符

* \<start>表示一个句子的开始
* \<end>表示一个句子的结束
* \<unknown>表示训练集中未见过的字符

然后形成一个词汇表，保存下来

#### data_load.py

写一个（或两个）coco数据集的pytorch的dataset格式

对于trani来说,要返回在一个mini-batch内按句子的长度排序的图片,句子，和句子的长度

对于val来说，为了方便，我们batch size取1，返回图片和图片的id

#### model.py

模型应该包括两个部分

* encoder我们采用预训练好的cnn（vgg16或resnet151)
* decoder应该包含以下两个方法：
  * forward，用来计算训练时每个时刻输出每个单词的概率。
  * generator，测试时用来生成真正的句子。

#### train_nic.py

训练的主程序之前在两个小实验中已经写过了，代码我也已经上传，大家可以参考。



#### utils

utils文件夹包含了一些要用的工具函数，我已经写好放了进去



#### Note：

不用服务器的同学，coco的评价指标需要自行安装，教程在https://github.com/flauted/coco-caption

用服务器的同学暂时还不能用测试的函数，我安装完会及时通知

update:

测试的函数已经可以使用，如果还是不能使用的话，请大家用189.240这台服务器。