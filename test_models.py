# test_models.py
# Kristy Duong <kristy5@cs.stanford.edu>

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import csv
import re
import math
import sys
import numpy as np

import util as U
import constants as C


input_size = 50
hidden_size = 1024
num_layers = 2
BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# sequence_length = 100
input_size = 50
hidden_size = 2048
num_layers = 2
num_epochs = 10
num_classes = 20
learning_rate = 0.00001

BATCH_SIZE = 64

class ClassBiRNN(nn.Module):
	def __init__(self, glove_vec, input_size, hidden_size, num_layers, num_classes):
		super(ClassBiRNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
		self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
		self.glove = glove_vec
		self.embed = nn.Embedding.from_pretrained(torch.tensor(glove_vec.vectors))
	
	def forward(self, x):
		# Set initial states
		embedding = self.embed(x) # B x L x E (batch x input_length x embedding_dim)
		h0 = torch.zeros(self.num_layers*2, embedding.size(0), self.hidden_size).to(device) # 2 for bidirection 
		c0 = torch.zeros(self.num_layers*2, embedding.size(0), self.hidden_size).to(device)
		# Forward propagate LSTM
		out, _ = self.lstm(embedding, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		# Decode the hidden state of the last time step
		out = self.fc(out[:, -1, :])
		# print(out)
		return out

# Regression model hyperparams
r_num_classes = 1

# Classification model hyperparams
c_num_classes = 20
# 
print("Loading GloVe vectors...")
glove = U.load_glove()

print("Reading in data...")
train_data, dev_data, test_data = U.get_data(glove = glove, device = device, classification = True)

# print("Loading regression model...")
# regression_model = BiRNN(glove, input_size, hidden_size, num_layers, r_num_classes).to(device)

print("Loading classification model...")
class_model = ClassBiRNN(glove, input_size, hidden_size, num_layers, c_num_classes).to(device)

# Test regression model
# output_path = C.filenames['bi_rnn']
# print("Testing regression model...")
# U.test_model(model, output_path, test_data, BATCH_SIZE, outputs = 'regression_results.txt')

# Test classification model

print("Testing classification on train data")
output_path = C.filenames['class_rnn']
U.test_model(class_model, output_path, train_data, BATCH_SIZE, device, classification = True, outputs = 'classification_results_new_split_train.txt')


output_path = C.filenames['class_rnn']
print("Testing classification model on dev data")
U.test_model(class_model, output_path, dev_data, BATCH_SIZE, device, classification = True, outputs = 'classification_results_new_split_dev.txt')


output_path = C.filenames['class_rnn']
print("Testing classification model... on test data")
U.test_model(class_model, output_path, test_data, BATCH_SIZE, device, classification = True, outputs = 'classification_results_new_split_test.txt')