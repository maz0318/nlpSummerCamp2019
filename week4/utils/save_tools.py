import pandas as pd
import os
import json
import torch

from utils.general_tools import *


def save_loss(epoch_loss, epoch, loss_path):
	'''
	Save loss information
	:param epoch_loss: Current epoch loss
	:param epoch: Current epoch
	:param loss_path: Path to save loss file
	:return:
	'''

	if epoch == 1:
		loss_file = pd.DataFrame(columns=['epoch', 'loss'])
		loss_file.to_csv(loss_path, index=0)

	loss_file = pd.read_csv(loss_path)

	loss_file = loss_file.append({'epoch': epoch, 'loss': epoch_loss}, ignore_index=True)

	loss_file.to_csv(loss_path, index=0)

	print("*** Epoch:{},Loss:{} ***".format(epoch, epoch_loss))


def save_generated_captions(generated_captions, epoch, generated_captions_folder_path, fine_tuning):
	'''
	Save validation generated captions of every epoch
	:param generated_captions: Generated captions
	:param epoch: Current epoch
	:param generated_captions_folder_path: Folder path to save every epoch generated captions
	:return: generated captions path
	'''

	if not os.path.exists(generated_captions_folder_path):
		os.mkdir(generated_captions_folder_path)
	if fine_tuning == False:
		json_path = generated_captions_folder_path + '{}.json'.format(epoch)
	else:
		json_path = generated_captions_folder_path + 'fine_tuning_{}.json'.format(epoch)

	with open(json_path, 'w') as f:
		json.dump(generated_captions, f)

	return json_path


def save_metrics(results, epoch, metrics_path):
	'''

	:param result: Current epoch results
	:param metrics_result_path:
	:return:
	'''
	if epoch == 1:
		metrics_list = ['epoch', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
		metrics_file = pd.DataFrame(columns=metrics_list)
		metrics_file.to_csv(metrics_path, index=0)

	metrics_file = pd.read_csv(metrics_path)
	score_dict = {'epoch': epoch}
	for metric, score in results:
		print("*** {}:{} ***".format(metric, round(score, 3)))
		score_dict[metric] = round(score, 3)
	metrics_file = metrics_file.append(score_dict, ignore_index=True)
	metrics_file.to_csv(metrics_path, index=0)
	return score_dict['CIDEr']


def save_best_model(encoder, decoder, optimizer, epoch, best_score, best_epoch, best_model_path):
	torch.save({
		'encoder': encoder.state_dict(),
		'decoder': decoder.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch,
		'best_score': best_score,
		'best_epoch': best_epoch,
	}, best_model_path)


def save_epoch_model(encoder, decoder, optimizer, epoch, best_score, best_epoch, epoch_model_path, fine_tuning):
	if not os.path.exists(epoch_model_path):
		os.mkdir(epoch_model_path)
	if fine_tuning == False:
		torch.save({
			'encoder': encoder.state_dict(),
			'decoder': decoder.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch,
			'best_score': best_score,
			'best_epoch': best_epoch,
		}, epoch_model_path + 'epoch_{}.tar'.format(epoch))
	else:
		torch.save({
			'encoder': encoder.state_dict(),
			'decoder': decoder.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch,
			'best_score': best_score,
			'best_epoch': best_epoch,
		}, epoch_model_path + 'fine_tuning_epoch_{}.tar'.format(epoch))


# def sentence_show(img_id, sentence, epoch,sentences_show_path):
#
# 	if not os.path.exists(sentences_show_path):
# 		os.mkdir(sentences_show_path)
# 	if epoch == 1
# 		img_df = pd.DataFrame(columns=['epoch',sentence])

if __name__ == '__main__':
	generate_captions = [{'image_id': 70, 'caption': 'a a a a a a '}, {'image_id': 68, 'caption': 'b b b b b b'}]
	generated_captions_path = save_generated_captions(generate_captions, 1, './generated_captions')
	results = coco_metrics(generated_captions_path, '/home/maz/Documents/data/coco/annotations/captions_val2014.json')
	save_metrics(results, './metrics.csv')
