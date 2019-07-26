import argparse
import torch
import os
import pickle
import random
import nltk
import json

import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

from model import *
from utils.general_tools import *
from utils.save_tools import *
from data_load import *
from pretreat import *

def set_seed(seed):
	'''
	Fix immediate seed to repeat experiment
	:param seed: An integer
	:return:
	'''
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

set_seed(21)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):

	# ==============================
	# Create some folders or files for saving
	# ==============================

	if not os.path.exists(args.root_folder.format(args.save_version)):
		os.mkdir(args.root_folder.format(args.save_version))

	loss_path = args.loss_path.format(args.save_version)
	mertics_path = args.mertics_path.format(args.save_version)
	epoch_model_path = args.epoch_model_path.format(args.save_version)
	best_model_path = args.best_model_path.format(args.save_version)
	generated_captions_path = args.generated_captions_folder_path.format(args.save_version)
	sentences_show_path = args.sentences_show_path.format(args.save_version)

	# Transform the format of images
	# This function in utils.general_tools.py
	train_transform = get_train_transform()
	val_transform = get_val_trainsform()

	# Load vocabulary
	print("*** Load Vocabulary ***")
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)

	# Create data sets
	# This function in data_load.py
	train_data = train_load(root=args.train_image_dir, json=args.train_caption_path, vocab=vocab,
						   transform=train_transform,
						   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

	val_data = val_load(root=args.val_image_dir, json=args.val_caption_path,
						 transform=val_transform,
						 batch_size=1, shuffle=False, num_workers=args.num_workers)

	# Build model
	encoder = Encoder(args.hidden_dim,args.fine_tuning).to(device)
	decoder = Decoder(args.embedding_dim, args.hidden_dim, vocab, len(vocab), args.max_seq_length).to(device)

	# Select loss function
	criterion = nn.CrossEntropyLoss().to(device)

	if args.fine_tuning == True:
		params = list(decoder.parameters()) + list(encoder.parameters())
		optimizer = torch.optim.Adam(params,lr=args.fine_tuning_lr)
	else:
		params = decoder.parameters()
		optimizer = torch.optim.Adam(params, lr=args.lr)

	# Load pretrained model
	if args.resume == True:
		checkpoint = torch.load(best_model_path)
		encoder.load_state_dict(checkpoint['encoder'])
		decoder.load_state_dict(checkpoint['decoder'])
		if args.fine_tuning == False:
			optimizer.load_state_dict(checkpoint['optimizer'])
		start_epoch = checkpoint['epoch'] + 1
		best_score = checkpoint['best_score']
		best_epoch = checkpoint['best_epoch']

	# New epoch and score
	else:
		start_epoch = 1
		best_score = 0
		best_epoch = 0

	for epoch in range(start_epoch, 10000):

		print("-" * 20)
		print("epoch:{}".format(epoch))

		# Adjust learning rate when the difference between epoch and best epoch is multiple of 3
		if (epoch - best_epoch) > 0 and (epoch - best_epoch) % 4 == 0:
			# This function in utils.general_tools.py
			adjust_lr(optimizer, args.shrink_factor)
		if (epoch - best_epoch) > 10 :
			break
			print("*** Training complete ***")

		# =============
		# Training
		# =============

		print(" *** Training ***")
		decoder.train()
		encoder.train()
		total_step = len(train_data)
		epoch_loss = 0
		for (images, captions, lengths, img_ids) in tqdm(train_data):
			images = images.to(device)
			captions = captions.to(device)
			# Why do lengths cut 1 and the first dimension of captions from 1
			# Because we need to ignore the begining symbol <start>
			lengths = list(np.array(lengths) - 1)

			targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
			features = encoder(images)
			predictions = decoder(features, captions, lengths)
			predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]

			loss = criterion(predictions, targets)
			epoch_loss += loss.item()
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()


		# Save loss information
		# This function in utils.save_tools.py
		save_loss(round(epoch_loss / total_step, 3), epoch, loss_path)


		# =============
		# Evaluating
		# =============

		print("*** Evaluating ***")
		encoder.eval()
		decoder.eval()
		generated_captions = []
		for image, img_id in tqdm(val_data):

			image = image.to(device)
			img_id = img_id[0]

			features = encoder(image)
			sentence = decoder.generate(features)
			sentence = ' '.join(sentence)
			item = {'image_id': int(img_id), 'caption': sentence}
			generated_captions.append(item)
			j = random.randint(1,100)


		print('*** Computing metrics ***')

		# Save current generated captions
		# This function in utils.save_tools.py

		captions_json_path = save_generated_captions(generated_captions, epoch, generated_captions_path,args.fine_tuning)

		# Compute score of metrics
		# This function in utils.general_tools.py
		results = coco_metrics(args.val_caption_path, captions_json_path, epoch, sentences_show_path)

		# Save metrics results
		# This function in utils.save_tools.py
		epoch_score = save_metrics(results, epoch, mertics_path)

		# Update the best score
		if best_score < epoch_score:

			best_score = epoch_score
			best_epoch = epoch

			save_best_model(encoder, decoder, optimizer, epoch, best_score, best_epoch, best_model_path)

		print("*** Best score:{} Best epoch:{} ***".format(best_score, best_epoch))
		# Save every epoch model
		save_epoch_model(encoder, decoder, optimizer, epoch, best_score, best_epoch, epoch_model_path,args.fine_tuning)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# =================
	# Load parameter
	# =================
	parser.add_argument('--vocab_path', type=str,
						default='/home/maz/Documents/data/coco/annotations/vocab.pkl',
						help="Storage path of vocabulary")

	parser.add_argument('--train_image_dir', type=str,
						default='/home/maz/Documents/data/coco/coco_image2014',
						help="Image path of training data set")

	parser.add_argument('--train_caption_path', type=str,
						default='/home/maz/Documents/data/coco/annotations/captions_train2014.json',
						help="Caption path of training data set")

	parser.add_argument('--val_image_dir', type=str,
						default='/home/maz/Documents/data/coco/val2014',
						help="Image path of validation set")

	parser.add_argument('--val_caption_path', type=str,
						default='/home/maz/Documents/data/coco/annotations/captions_val2014.json',
						help="Caption path of validation set")

	parser.add_argument('--fine_tuning', type=bool,
						default=False,
						help="fine tuning model")

	parser.add_argument('--resume', type=bool,
						default=False,
						help="Continue a pretrained model")

	# ================
	# Save parameter
	# ================

	parser.add_argument('--root_folder', type=str,
						default='../../log/caption/nic/{}/',
						help="Root directory of log information")

	parser.add_argument('--save_version', type=str,
						default='res152',
						help="Distinguish different saved information ")

	parser.add_argument('--loss_path', type=str,
						default='../../log/caption/nic/{}/loss.csv',
						help="Path to save loss information file")

	parser.add_argument('--mertics_path', type=str,
						default='../../log/caption/nic/{}/metrics_result.csv',
						help="Path to save metrics result file")

	parser.add_argument('--epoch_model_path', type=str,
						default='../../log/caption/nic/{}/epoch_model/',
						help="Folder Path to save every epoch model weights file")

	parser.add_argument('--best_model_path', type=str,
						default='../../log/caption/nic/{}/best_model.tar',
						help="Path to save best model weights file")

	parser.add_argument('--generated_captions_folder_path', type=str,
						default='../../log/caption/nic/{}/generated_captions/',
						help="Folder Path to save every epoch generated_captions")

	parser.add_argument('--sentences_show_path', type=str,
						default='../../log/caption/nic/{}/sentences_show/',
						help="To show generated sentence of every image in every epoch, including all metrics")

	# =================
	# model parameter #
	# =================

	parser.add_argument('--batch_size', type=int,
						default=64,
						help="Size of a mini-batch")

	parser.add_argument('--num_workers', type=str,
						default=1,
						help="Number of threads reading data")

	parser.add_argument('--embedding_dim', type=int,
						default=512,
						help="Dimention of word embedding")

	parser.add_argument('--hidden_dim', type=int,
						default=512,
						help=" Number of hidden layer cells in LSTM")

	parser.add_argument('--max_seq_length', type=int,
						default=20,
						help='Maximum length of prediction sequence')

	parser.add_argument('--lr', type=float,
						default=4e-4,
						help='Learning rate')

	parser.add_argument('--fine_tuning_lr', type=float,
						default=1e-4,
						help='Learning rate of fine tuning')

	parser.add_argument('--shrink_factor', type=float,
						default=0.8,
						help='Decay factor of learning rate')

	args = parser.parse_args()
	main(args)
