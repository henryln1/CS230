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

BATCH_SIZE = 1

# train_images, train_labels = generate_minibatch.export_main(input_size, batch_size)
# EXAMPLE_LENGTH = 100

# glove_vectors = generate_minibatch.export_glove(input_size)
# glove_vector_dim = len(glove_vectors['the'])

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

# # Read in dev/test data
# test_images = None
# test_labels = None
# with open('dev_data.csv', encoding="utf8") as file:
#   reader = csv.reader(file, delimiter=',')
#   num_test_examples = sum(1 for row in reader)

# with open('dev_data.csv', encoding="utf8") as file:
#   reader = csv.reader(file, delimiter=',')
#   test_images = np.zeros(shape = (num_test_examples - 1, EXAMPLE_LENGTH, glove_vector_dim))
#   test_labels = np.zeros(shape = (num_test_examples - 1, 1))
#   for idx, example in enumerate(reader):
#     if idx == 0:
#       continue
#     _, subreddit, title, score, num_comments, timestamp = tuple(example)  
#     title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()
#     title_words_list = [x.lower() for x in title_words_list]
#     for j in range(len(title_words_list)):
#       curr_word = title_words_list[j]
#       if curr_word in glove_vectors:
#         test_images[idx-1][j] = glove_vectors[curr_word]
#         test_labels[idx-1] = float(score)

# # train_images, train_labels = generate_minibatch.export_main(input_size, batch_size)
# train_images = torch.from_numpy(train_images)
# train_labels = torch.from_numpy(train_labels)

# test_images, test_labels = generate_minibatch.export_main(input_size, batch_size)
# # test_images, test_labels = generate_minibatch.export_main(input_size, batch_size)
# test_images = torch.from_numpy(test_images)
# test_labels = torch.from_numpy(test_labels)

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
		# self.lstm.flatten_parameters()
		out, _ = self.lstm(embedding, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		# print(out.shape)
		# print(out)
		# print(out[:, -1, :])
		# Decode the hidden state of the last time step
		out = self.fc(out[:, -1, :])
		# out = self.fc(out[0])
		# print(out)
		return out.view(-1)

# Load GloVe vectors

print("Loading GloVe vectors...")
glove = U.load_glove()
print("Reading in data...")
# Read in data
train_data, dev_data, test_data = U.get_data(glove = glove, device = device)
print("Loading model...")
# Load model
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
		if(len(train_x) == 0):
			continue
		logits = model.forward(train_x.to(device))
		loss = loss_fn(logits, train_y.type('torch.FloatTensor').to(device))
		loss.backward()
		optimizer.step()

		# print("Logits: {}".format(logits))
		total_loss += loss.item()
		idxs = logits
		# idxs = torch.argmax(logits, dim=1)
		num_correct += torch.sum(idxs == train_y.type('torch.FloatTensor').to(device)).item()
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
		logits = model.forward(dev_x.to(device))
		total_dev_loss += loss_fn(logits, dev_y.type('torch.FloatTensor').to(device)).item()
		idxs = torch.argmax(logits, dim=1)
		num_correct += torch.sum(idxs == dev_y.type('torch.FloatTensor').to(device)).item()
	dev_acc = num_correct / len(dev_data[0])

	print("Dev loss is {}".format(total_dev_loss))
	print("Dev Accuracy: {}".format(dev_acc))
	# return total_loss / num_updates, total_dev_loss, dev_acc, batches

	# for i in range(len(train_images)):
	# # for i, (images, labels) in enumerate(train_loader):
	#     images = train_images[i].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
	#     labels = train_labels[i].type('torch.FloatTensor').to(device)
		  
	#     # Forward pass
	#     outputs = model(images)
	#     loss = loss_fn(outputs, labels)
		
	#     # Backward and optimize
	#     optimizer.zero_grad()
	#     loss.backward()
	#     optimizer.step()
		
	#     if (i+1) % 100 == 0:
	#         print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
	#                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test the model
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for i in range(len(test_labels)):
#     # for images, labels in test_loader:
#         images = test_images[i].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
#         labels = test_labels[i].type('torch.LongTensor').to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Test Accuracy of the model on the 100 test examples: {} %'.format(100 * correct / total)) 
# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')