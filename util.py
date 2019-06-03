# util.py
# Kristy Duong <kristy5@cs.stanford.edu>

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import torch
import sys
import os
import re

import constants as C

def plot_losses(train_losses, dev_losses, output_file=None):
	plt.plot(range(len(train_losses)), train_losses)
	plt.plot(range(len(dev_losses)), dev_losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(('Train Loss', 'Dev Loss'))
	if output_file == None:
		output_file = 'plots/graph.png'
	elif output_file[-3:] is not 'png':
		output_file += '.png'
	plt.savefig(output_file)
	print('Saved graph!')

def process_as_classification(label):
	upper_bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	for idx in range(1, len(upper_bounds)):
		if label in range(upper_bounds[idx - 1], upper_bounds[idx]):
			return idx - 1
	if label >= 1000:
		return len(upper_bounds)

def read_corpus(file_path, word_vectors = None, device = None, classification = False):
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
			if classification:
				labels.append(process_as_classification(int(score)))
			else:
				labels.append(int(score))
	labels = torch.tensor(labels).to(device)
	return data, labels

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

# Shuffle data
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