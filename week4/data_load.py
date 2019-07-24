import torch
import os
import nltk

import torch.utils.data as data

from PIL import Image
from pycocotools.coco import COCO

import pickle
import torchvision.transforms as transforms
from pretreat import Vocabulary


class CocoTrainset(data.Dataset):

	def __init__(self, root, json, vocab, transform=None):
		'''

		:param root: Images path
		:param json: Captions path
		:param vocab: Vocabulary
		:param transform: Process images
		:return:
		'''

		self.root = root
		self.coco = COCO(json)
		self.ids = list(self.coco.anns.keys())
		self.vocab = vocab
		self.transform = transform

	def __getitem__(self, index):
		'''

		return a item
		'''
		coco = self.coco
		vocab = self.vocab
		ann_id = self.ids[index]

		caption = coco.anns[ann_id]['caption']
		img_id = coco.anns[ann_id]['image_id']
		path = coco.loadImgs(img_id)[0]['file_name']

		image = Image.open(os.path.join(self.root, path)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = []
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
		return image, target, img_id

	def __len__(self):
		return len(self.ids)


def train_collate_fn(data):
	'''

	:param data: -format:(image,caption,img_id)
	:return: images: tensor (batch_size,3,224,224)
			 targets: tensor (batch_size,padded_length)
			 lenghts: list,Every effective length of padding caption
			 img_ids:list,id of image
	'''
	# Sort by captions' length
	data.sort(key=lambda x: len(x[1]), reverse=True)

	images, captions, img_ids = zip(*data)

	images = torch.stack(images, 0)

	lengths = [len(cap) for cap in captions]

	targets = torch.zeros(len(captions), max(lengths)).long()

	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]

	return images, targets, lengths, list(img_ids)


def train_load(root, json, vocab, transform, batch_size, shuffle, num_workers):
	coco = CocoTrainset(root=root, json=json, vocab=vocab, transform=transform)

	data_loader = data.DataLoader(dataset=coco,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers,
								  collate_fn=train_collate_fn,
								  drop_last=True)

	return data_loader




class CocoValset(data.Dataset):

	def __init__(self, root, json, transform=None):
		'''

		:param root: Images path
		:param json: Captions path
		:param vocab: Vocabulary
		:param transform: Process images
		:return:
		'''

		self.root = root
		self.coco = COCO(json)
		self.ids = list(self.coco.imgs.keys())
		self.transform = transform

	def __getitem__(self, index):
		'''
		return a item
		'''
		coco = self.coco
		img_id = self.ids[index]

		path = coco.loadImgs(img_id)[0]['file_name']

		image = Image.open(os.path.join(self.root, path)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		return image, img_id

	def __len__(self):
		return len(self.ids)


def val_load(root, json, transform, batch_size, shuffle, num_workers):
	coco = CocoValset(root=root, json=json, transform=transform)

	data_loader = data.DataLoader(dataset=coco,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers,
								  )

	return data_loader




# if __name__ == '__main__':
#
#     root = '/home/maz//data/coco/train2014'
#     root = '/home/maz/Documents/data/coco/train2014'
#     json = '/home/maz//data/coco/annotations/captions_train2014.json'
#     json = '/home/maz/Documents/data/coco/annotations/captions_train2014.json'
#     vocab_path = '/home/maz//data/coco/annotations/vocab.pkl'
#     vocab_path = '/home/maz/Documents/data/coco/annotations/vocab.pkl'
#     with open(vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
#     transform = transforms.Compose([
#         transforms.Resize([300, 300], Image.ANTIALIAS),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])
#     batch_size = 2
#     shuffle = False
#     num_workers = 1
#
#     dataset = data_load(root,json,vocab,transform,batch_size,shuffle,num_workers)
#     for a,b,c,d in dataset:
#         print(d)
#
#         break
