import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# import pickle

# class Vocabulary(object):
#
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = {}
#         self.idx = 0
#
#     def add_word(self,word):
#         if not word in self.word2idx:
#             self.word2idx[word] = self.idx
#             self.idx2word[self.idx] = word
#             self.idx+=1
#
#     def __call__(self,word):
#
#         # If a word not in vocabulary,it will be replace by <unknown>
#         if not word in self.word2idx:
#             return self.word2idx['<unk>']
#         return self.word2idx[word]
#
#     def __len__(self):
#         return len(self.word2idx)

class Encoder(nn.Module):

	def __init__(self,hidden_dim,fine_tuning):
		super(Encoder, self).__init__()

		cnn = models.resnet152(pretrained=True)
		modules = list(cnn.children())[:-2]
		self.cnn = nn.Sequential(*modules)
		self.affine_1 = nn.Linear(512, hidden_dim)
		for p in self.cnn.parameters():
			p.requires_grad = False
		if fine_tuning == True:
			self.fine_tune(fine_tuning=fine_tuning)

	def forward(self, images):

		features = self.cnn(images)
		features = features.permute(0, 2, 3, 1)
		features = features.reshape(features.size(0), -1,512)
		features = self.affine_1(features)
		return features

	def fine_tune(self, fine_tuning=False):
		for c in list(self.cnn.children())[7:]:
			for p in c.parameters():
				p.requires_grad = fine_tuning

class Decoder(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab, vocab_size, max_seq_length):

		super(Decoder, self).__init__()
		self.vocab_size = vocab_size
		self.vocab = vocab
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstmcell = nn.LSTMCell(embedding_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, vocab_size)
		self.max_seq_length = max_seq_length
		self.init_h = nn.Linear(512, hidden_dim)
		self.init_c = nn.Linear(512, hidden_dim)

	def forward(self, features, captions, lengths, state=None):

		batch_size = features.size(0)
		vocab_size = self.vocab_size
		embeddings = self.embedding(captions)
		predictions = torch.zeros(batch_size, max(lengths), vocab_size).to(device)
		h, c = self.init_hidden_state(features)

		for t in range(max(lengths)):
			batch_size_t = sum([l > t for l in lengths])
			h, c = self.lstmcell(embeddings[:batch_size_t, t, :],
								 (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t,hidden_dim)
			preds = self.fc(h)
			predictions[:batch_size_t, t, :] = preds

		return predictions

	def generate(self, features, state=None):

		sentence = []
		h, c = self.init_hidden_state(features)
		input = self.embedding(torch.tensor([1]).to(device))

		for i in range(self.max_seq_length):

			h, c = self.lstmcell(input, (h, c))
			prediction = self.fc(h)
			_, prediction = prediction.max(1)
			word = self.vocab.idx2word[int(prediction)]
			if word == '<end>':
				break
			sentence.append(word)
			input = self.embedding(prediction)

		return sentence


	def init_hidden_state(self, features):
		mean_features = features.mean(dim=1)
		h = self.init_h(mean_features)
		c = self.init_c(mean_features)
		return h, c
# if __name__ == '__main__':
#
# 	embedding_dim = 512
# 	hidden_dim = 512
# 	vocab_path = '/home/maz/文档/data/coco/annotations/vocab.pkl'
# 	with open(vocab_path, 'rb') as f:
# 		vocab = pickle.load(f)
# 	vocab_size = len(vocab)
# 	encoder = Encoder(embedding_dim)
# 	decoder = Decoder(embedding_dim,hidden_dim,vocab,vocab_size)
#
# 	samples = torch.Tensor(1,3,224,224)
# 	captions = torch.tensor([[1,2,3,4,5,6,7],[1,7,6,5,4,3,0]])
# 	lengths = [7,6]
# 	features = encoder.forward(samples,False)
# 	# pred = decoder.forward(features, captions, lengths)
# 	pred = decoder.generate(features)
# 	print(pred)
