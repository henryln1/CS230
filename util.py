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
import math


import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import constants as C


def evaluate_model(model, data, batch_size, device, classification = False, outputs = None):
	model.eval()
	n_minibatches = math.ceil(len(data[0]) / batch_size)
	batches = []
	total_loss = 0.
	num_correct = 0
	predicted = []
	actual = []
	for i in range(n_minibatches):
		start = i * batch_size
		end = start + batch_size
		batches.append((
			nn.utils.rnn.pad_sequence(
				data[0][start:end]).transpose(0, 1),
			data[1][start:end]))
	print("Device:", device)
	for batch in batches:
		x, y = batch		
		if(x.size(1) == 0):
			continue
		logits = model.forward(x.to(device))
		if classification:
			loss_fn = nn.CrossEntropyLoss()
			total_loss += loss_fn(logits, y.to(device)).item()
			# total_loss += nn.CrossEntropyLoss()(logits, y.to(device)).item()
			idxs = torch.argmax(logits, dim=1)
			num_correct += torch.sum(idxs == y.to(device)).item()
		else:
			total_loss += nn.SmoothL1Loss()(logits, y.type('torch.FloatTensor').to(device)).item()
			idxs = logits
			num_correct += torch.sum(idxs == torch.round(y.type('torch.FloatTensor')).to(device)).item()

		predicted += idxs.tolist()
		actual += y.tolist()
	accuracy = num_correct / len(data[0])

	print("Test Loss: {}".format(total_loss))
	print("Accuracy: {}".format(accuracy))
	with open(outputs, 'w') as f:
		f.write('Predicted\n' + str(predicted) + '\n')
		f.write('Actual\n ' + str(actual) + '\n')
	f.close()
	return total_loss, accuracy

def test_model(model, output_path, test_data, batch_size, device, classification = False, outputs = None):
	print(80 * "=")
	print("Testing")
	print(80 * "=")
	print("Restoring the best model weights found on the dev set...")
	model_dict = torch.load(output_path)
	model.load_state_dict(model_dict)
	print("Final evaluation on test set")
	print("Evaluating model on test data...")
	loss, accuracy = evaluate_model(model, test_data, batch_size, device, classification = True, outputs = outputs)

	return loss, accuracy

def plot_all_losses(train_losses):
	plt.plot(range(len(train_losses)), train_losses)
	print("Length of train losses:", len(train_losses))
	print("Batch Loss: ", train_losses)
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.title('Loss Curve')
	plt.savefig('graph_batch_loss_06_07_19_classification_v2_low_lr_new_split.png')
	plt.close()

def plot_losses(train_losses, dev_losses):
	plt.plot(range(len(train_losses)), train_losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Loss')
	# plt.legend(('Train Loss', 'Dev Loss'))
	plt.savefig("graph_train_loss_06_07_19_classification_v2_low_lr_new_split.png")
	plt.close()
	plt.plot(range(len(dev_losses)), dev_losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Dev Loss')
	plt.savefig("graph_dev_loss_06_07_19_classification_v2_low_lr_new_split.png")
	plt.close()
	print('Saved graph!')

def process_as_classification(label):
	"""
	Processes Reddit score into classification range
	@param label (int): Reddit score
	@return: classification label
	"""
	# print("Label: ", label)
	# upper_bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	upper_bounds = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	for idx in range(1, len(upper_bounds)):
		if label in range(upper_bounds[idx - 1], upper_bounds[idx]):
			# print("IDX: ", idx - 1)
			return idx - 1
	if label >= 1000:
		# print("IDX: ", len(upper_bounds) - 1)
		return len(upper_bounds) - 1


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

def get_data(glove = None, device = None, classification = False):
	pickle_name = C.filenames['pickle']
	train_file_name = C.filenames['train']
	dev_file_name = C.filenames['dev']
	test_file_name = C.filenames['test']

	if classification:
		pickle_name = C.filenames['classification_pickle']

	try:
		data = load_large_pickle(pickle_name)
		print('Continuing with loaded pickle for padding data...')
	except Exception as e:
		print('Unable to find pre-processed data. Pre-processing new...')
		train_data, train_labels = read_corpus(train_file_name, glove, device, classification)
		full_train_data = [train_data, train_labels]
		shuffle_data(full_train_data)

		dev_data, dev_labels = read_corpus(dev_file_name, glove, device, classification)
		full_dev_data = [dev_data, dev_labels]
		shuffle_data(full_dev_data)

		test_data, test_labels = read_corpus(test_file_name, glove, device, classification)
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
