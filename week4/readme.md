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