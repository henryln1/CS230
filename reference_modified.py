# https://github.com/yunjey/pytorch-tutorial
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# import generate_minibatch
import csv
import re
import math
import numpy as np

import util as U

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# sequence_length = 100
input_size = 50
hidden_size = 128
num_layers = 2
num_epochs = 2
num_classes = 1
learning_rate = 0.001

BATCH_SIZE = 64

class BiRNN(nn.Module):
	def __init__(self, glove_vec, input_size, hidden_size, num_layers, num_classes):
		super(BiRNN, self).__init__()
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
		return out.view(-1)

print("Loading GloVe vectors...")
glove = U.load_glove()

print("Reading in data...")
train_data, dev_data, test_data = U.get_data(glove = glove, device = device)

print("Loading model...")
model = BiRNN(glove, input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
# Train the model
for epoch in range(num_epochs):
	print("Epoch {:} out of {:}".format(epoch + 1, num_epochs))

	model.train()

	n_minibatches = math.ceil(len(train_data[0]) / BATCH_SIZE)

	total_loss = 0.
	num_updates = 0

	# Create minibatches for train data
	# if batches == None:
	batches = []
	for i in range(n_minibatches):
		start = i * BATCH_SIZE
		end = start + BATCH_SIZE
		batches.append((
			nn.utils.rnn.pad_sequence(
				train_data[0][start:end]).transpose(0, 1),
			train_data[1][start:end]))

	# loss_meter = AverageMeter()

	num_correct = 0
	# Train model
	for batch in batches:
		optimizer.zero_grad()
		loss = 0.
		train_x, train_y = batch
		if(train_x.size(1) == 0):
			continue
		logits = model.forward(train_x.to(device))
		loss = loss_fn(logits, train_y.type('torch.FloatTensor').to(device))
		loss.backward()
		optimizer.step()

		# print("Logits: {}".format(logits))
		total_loss += loss.item()
		idxs = logits
		# idxs = torch.argmax(logits, dim=1)
		num_correct += torch.sum(idxs == torch.round(train_y.type('torch.FloatTensor')).to(device)).item()
		print("Predictions: {}".format(idxs))
		print("Actual: {}".format(train_y))
		print("Loss: {}".format(loss) )
	train_acc = num_correct / len(train_data[0])
		# num_updates += 1

		# loss_meter.update(loss.item())

	print("Train loss is {}".format(total_loss))
	print("Train Accuracy: {}".format(train_acc))

	model.eval()
	dev_n_minibatches = math.ceil((len(dev_data[0]) - 1) / BATCH_SIZE)
	dev_batches = []
	total_dev_loss = 0.
	num_correct = 0

	# Create minibatches for dev data
	for i in range(dev_n_minibatches):
		start = i * BATCH_SIZE
		end = start + BATCH_SIZE
		dev_batches.append((
			nn.utils.rnn.pad_sequence(
				dev_data[0][start:end]).transpose(0, 1),
			dev_data[1][start:end]))

	# Evaulate model on dev data
	for batch in dev_batches:
		dev_x, dev_y = batch
		if(dev_x.size(1) == 0):
			continue
		logits = model.forward(dev_x.to(device))
		total_dev_loss += loss_fn(logits, dev_y.type('torch.FloatTensor').to(device)).item()
		idxs = logits
		num_correct += torch.sum(idxs == torch.round(dev_y.type('torch.FloatTensor')).to(device)).item()
	dev_acc = num_correct / len(dev_data[0])

	print("Dev loss is {}".format(total_dev_loss))
	print("Dev Accuracy: {}".format(dev_acc))
	# return total_loss / num_updates, total_dev_loss, dev_acc, batches