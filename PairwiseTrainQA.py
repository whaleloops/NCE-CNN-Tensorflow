# On-going construction
# NON-debugged code

import argparse
import sys
import numpy as np

from util.Vocab import Vocab
from util.read_data import read_relatedness_dataset, read_embedding

#%% functions
def load_glove_word_embeddings(vocab):
	emb_dir = 'data/glove/'
	emb_prefix = emb_dir + 'glove.840B'
	emb_vocab, emb_vecs = read_embedding(emb_prefix + '.vocab', emb_prefix + '.300d.npy')
	emb_dim = emb_vecs.shape[1]

	# use only vectors in vocabulary (not necessary, but gives faster training)

	num_unk = 0
	vecs = np.zeros([vocab.size, emb_dim])

	UNK = np.random.uniform(low=-0.05, high=0.05, size=emb_dim)

	for i in range(0, vocab.size):
		w = vocab.token(i)
		if emb_vocab.contains(w):
			vecs[i] = emb_vecs[emb_vocab.index(w)]
		else:
			vecs[i] = emb_vecs[emb_vocab.index('unk')] #UNK --:uniform(-0.05, 0.05)
			num_unk = num_unk + 1
	print('unk count = %d' % num_unk)
	return vecs

#%% main
if __name__ == '__main__':
	# Configure the argument parser
	parser = argparse.ArgumentParser(description = 'Python to run pairwise training for QA')
	parser.add_argument('--dataset', action='store', dest='dataset', default='TrecQA',
	                    help='dataset, can be TrecQA or WikiQA')
	parser.add_argument('--version', action='store', dest='version', default='raw',
	                    help='the version of TrecQA dataset, can be raw and clean')
	parser.add_argument('--num_pairs', action='store', dest='num_pairs', type=int, default=8,
	                    help='number of negative samples for each pos sample')
	parser.add_argument('--neg_mode', action='store', dest='neg_mode', type=int, default=2,
	                    help='negative sample strategy, 1 is random sampling, 2 ismax sampling and 3 is mix sampling')
	opt = parser.parse_args()
	# pid = arguments.a

	# read default arguments 
	args = {
	  'model' : 'pairwise-conv', #convolutional neural network 
	  'layers' : 1, # number of hidden layers in the fully-connected layer
	  'dim' : 150, # number of neurons in the hidden layer.
	  'dropout_mode' : 1 # add dropout by default, to turn off change its value to 0
	}

	model_name = 'pairwise-conv'
	# model_class = similarityMeasure.Conv # TODO class name
	model_structure = model_name

	# torch.seed() TODO0
	# torch.manualSeed(-3.0753778015266e+18)
	# print('<torch> using the automatic seed: ' .. torch.initialSeed())

	if opt.dataset != 'TrecQA' and opt.dataset != 'WikiQA':
		print('Error dataset!')
		sys.exit()

	# directory containing dataset files
	data_dir = 'data/' + opt.dataset + '/'

	# load vocab
	vocab = Vocab(data_dir + 'vocab.txt')

	# load embeddings
	print('loading glove word embeddings')
	vecs = load_glove_word_embeddings(vocab)
	taskD = 'qa'

	# load datasets
	print('loading datasets' + opt.dataset)
	if opt.dataset == 'TrecQA':
		train_dir = data_dir + 'train-all/'
		dev_dir = data_dir + opt.version + '-dev/'
		test_dir = data_dir + opt.version + '-test/'
	elif opt.dataset == 'WikiQA':
		train_dir = data_dir + 'train/'
		dev_dir = data_dir + 'dev/'
		test_dir = data_dir + 'test/'

	# train_dataset = read_relatedness_dataset(train_dir, vocab, taskD) # This is a dict
	# dev_dataset = read_relatedness_dataset(dev_dir, vocab, taskD)
	test_dataset = read_relatedness_dataset(test_dir, vocab, taskD)
	# print('train_dir: %s, num train = %d\n' % (train_dir, train_dataset['size']))
	# print('dev_dir: %s, num dev   = %d\n' % (dev_dir, dev_dataset['size']))
	print('test_dir: %s, num test  = %d\n' % (test_dir, test_dataset['size']))
	
	# extra data checkings:
	t1 = test_dataset
	print('train_dataset.size = %d' % t1['size']) 
	print('#train_dataset.lsents = %d' %  len(t1['lsents']))
	print(t1['lsents'][1])
	print('#train_dataset.rsents = %d' %  len(t1['rsents']))
	print(t1['rsents'][1])
	print('#train_dataset.ids = %d' %  len(t1['ids']))
	print(t1['ids'][1])
	print('--#train_dataset.boundary = %d' %  len(t1['boundary']))
	print(t1['boundary'][1])
	print('--#train_dataset.numrels = %d' %  len(t1['numrels']))
	print(t1['numrels'][1])
	print('--#train_dataset.labels = %d' %  len(t1['labels']))
	print(t1['labels'][1])


	# initialize model TODO9:


