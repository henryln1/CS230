# util.py
# Kristy Duong <kristy5@stanford.edu>

import gensim.downloader as api
import numpy as np
import pickle
import csv
import torch
import sys
import os
import re

import constants as C

def read_corpus(file_path, word_vectors = None, device = None):
	""" Read file, where each sentence is dilineated by a `\n`.
	@param file_path (str): path to file containing corpus
	@return data: 2D list. First dimension is each line (example),
		second dimension is word index in glove embedding
	"""
	if word_vectors == None:
		word_vectors = load_glove()
	vocab = word_vectors.vocab
	data = []
	labels = []

	with open(file_path, encoding="utf8") as file:
		reader = csv.reader(file, delimiter=',')
		for idx, line in enumerate(reader):
			if idx == 0:
				continue
			_, subreddit, title, score, num_comments, timestamp = tuple(line)
			title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()    
			title_words_list = [x.lower() for x in title_words_list]
			if len(title_words_list) == 0:
				continue
			data.append(torch.LongTensor([vocab[word].index for word in title_words_list if word in word_vectors.vocab.keys()]).to(device))
			labels.append(int(score))
	labels = torch.tensor(labels).to(device)
	return data, labels

	# for line in open(file_path, encoding='latin-1'):
	# 	sent = line.strip().split(' ')
	# 	if len(sent) == 0:
	# 		continue
	# 	data.append(torch.LongTensor([vocab[word].index for word in sent if word in word_vectors.vocab.keys()]).to(device))
	# return data

def get_data(glove = None, device = None):
	pickle_name = C.filenames['pickle']
	train_file_name = C.filenames['train']
	dev_file_name = C.filenames['dev']
	test_file_name = C.filenames['test']

	try:
		data = load_large_pickle(pickle_name)
		print('Continuing with loaded pickle for padding data...')
	except Exception as e:
		print('Unable to find pre-processed data. Pre-processing new...')
		train_data, train_labels = read_corpus(train_file_name, glove, device)
		full_train_data = [train_data, train_labels]
		shuffle_data(full_train_data)

		dev_data, dev_labels = read_corpus(dev_file_name, glove, device)
		full_dev_data = [dev_data, dev_labels]
		shuffle_data(full_dev_data)

		test_data, test_labels = read_corpus(test_file_name, glove, device)
		full_test_data = [test_data, test_labels]
		shuffle_data(full_test_data)

		data = (full_train_data, full_dev_data, full_test_data)

		save_large_pickle(data, pickle_name)

	return data

# Read in train data
# train_images = None
# train_labels = None
# num_train_examples = 0
# with open('train_data.csv', encoding="utf8") as file:
#   reader = csv.reader(file, delimiter=',')
#   num_train_examples = sum(1 for row in reader)

# with open('train_data.csv', encoding="utf8") as file:
#   reader = csv.reader(file, delimiter=',')
#   train_images = np.zeros(shape = (num_train_examples - 1, EXAMPLE_LENGTH, glove_vector_dim))
#   train_labels = np.zeros(shape = (num_train_examples - 1, 1))
#   for idx, example in enumerate(reader):
#     if idx == 0:
#       continue
#     _, subreddit, title, score, num_comments, timestamp = tuple(example)  
#     title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()
#     title_words_list = [x.lower() for x in title_words_list]
#     for j in range(len(title_words_list)):
#       curr_word = title_words_list[j]
#       if curr_word in glove_vectors:
#         train_images[idx-1][j] = glove_vectors[curr_word]
#         train_labels[idx-1] = float(score)

# # train_images, train_labels = generate_minibatch.export_main(input_size, batch_size)
# train_images = torch.from_numpy(train_images)
# train_labels = torch.from_numpy(train_labels)

# test_images, test_labels = generate_minibatch.export_main(input_size, batch_size)
# # test_images, test_labels = generate_minibatch.export_main(input_size, batch_size)
# test_images = torch.from_numpy(test_images)
# test_labels = torch.from_numpy(test_labels)

def shuffle_data(data):
	data_len = len(data[0])
	new_indices = np.random.permutation(data_len)
	new_data = []
	for i in range(len(data)):
		try:
			data[i] = data[i][new_indices]
		except Exception as e:
			data[i] = [data[i][idx] for idx in new_indices]

# Load GloVe vectors
def load_glove():
	try:
		wv_from_bin = load_large_pickle(C.filenames['glove'])
		print('Continuing with loaded pickle for glove...')
	except Exception as e:
		print('Unable to find glove cache. Loading new...')
		wv_from_bin = api.load("glove-wiki-gigaword-50")
		vocab = list(wv_from_bin.vocab.keys())
		print("Loaded vocab size %i" % len(vocab))
		save_large_pickle(wv_from_bin, C.filenames['glove'])
	return wv_from_bin

def save_large_pickle(obj, filepath):
	"""
	This is a defensive way to write pickle.write, allowing for very large files on all platforms
	"""
	max_bytes = 2**31 - 1
	bytes_out = pickle.dumps(obj)
	n_bytes = sys.getsizeof(bytes_out)
	with open(filepath, 'wb') as f_out:
		for idx in range(0, n_bytes, max_bytes):
			f_out.write(bytes_out[idx:idx+max_bytes])


def load_large_pickle(filepath):
	"""
	This is a defensive way to write pickle.load, allowing for very large files on all platforms
	"""
	max_bytes = 2**31 - 1
	input_size = os.path.getsize(filepath)
	bytes_in = bytearray(0)
	with open(filepath, 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	obj = pickle.loads(bytes_in)
	return obj