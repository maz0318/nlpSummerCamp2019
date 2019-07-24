import nltk
import pickle
import argparse

from collections import Counter
from pycocotools.coco import COCO
from tqdm import tqdm


class Vocabulary(object):

	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	def add_word(self, word):
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	def __call__(self, word):

		# If a word not in vocabulary,it will be replace by <unknown>
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)


def build_vocab(json, threshold):
	'''
	Bulid a vocabulary

	:param json: json of caption
	:param threshold: Only when frequency of a word is greater than threshold does it can be added in vocabulary
	:return: a vocabulary( pkl format )
	'''

	# Load captions
	coco = COCO(json)
	counter = Counter()
	ids = coco.anns.keys()

	for id in tqdm(ids):
		caption = str(coco.anns[id]['caption'])
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		counter.update(tokens)

	# Fillter the frequency is less than threshold
	words = [word for word, cnt in counter.items() if cnt >= threshold]

	# Build vocabulary
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	for word in words:
		vocab.add_word(word)
	return vocab


def main(args):
	vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
	vocab_path = args.vocab_path

	# save the vocabulary in pkl format
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)

	print("*** Vocabulary size：{} ***".format(len(vocab)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--caption_path', type=str,
						default='/home/maz/Documents/data/coco/annotations/captions_train2014.json',
						help='Annotation path of training set ')

	parser.add_argument('--vocab_path', type=str,
						default='/home/maz/Documents/data/coco/annotations/vocab.pkl',
						help='Storage path of vocabulary')

	parser.add_argument('--threshold', type=int,
						default=5,
						help='Minimum frequency of a word ')
	args = parser.parse_args()
	main(args)
