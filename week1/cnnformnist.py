'''
torch: 就是pytorch
torchvision: 是一个视觉的库，包括数据集，对图片处理的方式和遇训练好的模型
argparse: 统一配置超参数的库
os: 对系统的一些操作命令库
random：随机数库
numpy: 科学计算库
torch.nn:  里面包含了很多已经写好的神经网络
torchvision.transforms: 包含了很多对图片的处理
'''
import torch
import torchvision
import argparse
import os
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

def set_seed(seed):
	'''
	固定随机种子
	'''
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

set_seed(21)



# 选择cuda或者cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================
# 构建模型类（两层的卷积神经网络）
# =============================

class CNN(nn.Module):
	def __init__(self, out1_channel, out2_channel):
		super(CNN, self).__init__()

		#torch中的Sqeuential可以定义好一系列网络，它会按照顺序运行
		self.conv = nn.Sequential(
					# 卷积操作，参数分别为，input的channel维度，output的channel维度，卷积核的大小，卷积核移动步长，是否填充
					nn.Conv2d(1, out1_channel, 3,stride=1, padding=1),
					#最大值池化，图片像素会缩小4倍（28 * 28) -> (14 * 14)
					nn.MaxPool2d(2, stride=2),
					nn.ReLU(True),
					nn.Conv2d(out1_channel,out2_channel,3,stride=1, padding=1),
					nn.MaxPool2d(2, stride=2),

		)
		# 把所以像素拉成一个向量，经过两次池化，像素变成7*7，一共有20个channel，最后分成10类
		self.fc = nn.Linear(7*7*out2_channel,10)

	def forward(self, x):
		'''
		前向传播函数
		:param x: 数据
		:return:
		'''
		out = self.conv(x)
		out = out.reshape(out.size(0),-1)
		out = self.fc(out)
		return out


def main(agrs):

	# =============================
	# 构建数据集
	# =============================

	# MNIST dataset已经集成在torchvision.datasets中了，直接可以下载使用
	# transform.TOtensor()是使图片变成张量的形式
	train_dataset = torchvision.datasets.MNIST(root='../data',
											   train=True,
											   transform=transforms.ToTensor(),
											   download=True)

	test_dataset = torchvision.datasets.MNIST(root='../data',
											  train=False,
											  transform=transforms.ToTensor())

	# DataLoader是pytorch的一个数据读取的函数，用pytorch内置的dataset,或者按照datasets的格式写好
	# 可以方便的进行每一轮取多少个样本（batch_size)，打乱(shuffle)等操作
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=args.batch_size,
											   shuffle=True)

	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											  batch_size=args.batch_size,
											  shuffle=False)

	# 构建模型
	net = CNN(args.out1_channel, args.out2_channel).to(device)

	# 优化器 传入nn的参数，用梯度下降来更新参数
	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

	# 使用交叉熵作为损失函数
	loss_function = nn.CrossEntropyLoss().to(device)

	best_epoch = 0
	best_acc = 0

	print("*** Training ***")
	for epoch in range(1, 10000):

		print('-'*20)
		# 如果10轮效果都没有提升，就停止训练
		if epoch - best_epoch > 10:
			print("*** Training completing ***")
			break
		epoch_loss = 0.0
		total_step = len(train_loader)

		# =============================
		# 训练
		# =============================

		for images, labels in train_loader:

			batch_size = images.size(0)

			# 28*28的图片拉成一个784维的向量，-1表示全都拉成一个向量，也可以额输入784
			images = images.to(device)
			labels = labels.to(device)

			# 送入网络训练
			outputs = net(images)

			# 计算loss
			loss = loss_function(outputs, labels)
			epoch_loss += loss.item()
			# 清空梯度
			optimizer.zero_grad()
			# 回传loss
			loss.backward()
			# 优化器更新参数
			optimizer.step()
		print("*** epoch:{}, loss:{} ***".format(epoch,round(epoch_loss/total_step,3)))

		# =============================
		# 测试
		# =============================

		epoch_acc_num = 0
		epoch_num = 0
		for images, labels in test_loader:

			batch_size = images.size(0)

			# 28*28的图片拉成一个784维的向量，-1表示全都拉成一个向量，也可以额输入784
			images = images.to(device)
			labels = labels.to(device)

			outputs = net(images)
			# 预测结果
			_, predictions = torch.max(outputs,1)

			epoch_num += batch_size
			epoch_acc_num += (predictions == labels).sum().item()

		# 取3位小数
		epoch_acc = round(epoch_acc_num/epoch_num,3)
		print("*** epoch:{},acc:{} ***".format(epoch,epoch_acc))

		if epoch_acc > best_acc:
			best_acc = epoch_acc
			best_epoch = epoch

			model_path = './model_weight/'
			if not os.path.exists(model_path):
				os.mkdir(model_path)
			#保存效果最好的模型
			torch.save({
				'net':net.state_dict(),
				'optimizer':optimizer.state_dict(),
				'best_epoch':best_epoch,
				'best_acc':best_acc
			},model_path+'cnn_best_mode.tar')
	print("*** best_epoch:{}, best_acc:{}".format(best_epoch,best_acc))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# =================
	# Model parameter
	# =================

	parser.add_argument('--out1_channel',type=int,
						default=10,
						help="第一个卷积操作输出的通道数，可以改变")

	parser.add_argument('--out2_channel',type=int,
						default=100,
						help="第二个卷积操作输出的通道数，可以改变")

	parser.add_argument('--num_classes',type=int,
						default=10,
						help="mnist数据集最后的分类，默认10（0-9中数字)，不可改变")

	parser.add_argument('--lr',type=float,
						default=1e-3,
						help="模型的学习率")

	parser.add_argument('--batch_size', type=int,
						default=128,
						help="一次学习的样本数")

	args = parser.parse_args()
	main(args)