# https://github.com/yunjey/pytorch-tutorial
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import generate_minibatch
import csv
import numpy as np
import re

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# sequence_length = 28
sequence_length = 100
input_size = 50
# input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 1000
num_epochs = 2
learning_rate = 0.003

EXAMPLE_LENGTH = 100

glove_vectors = generate_minibatch.export_glove(input_size)
glove_vector_dim = len(glove_vectors['the'])

# Read in train data
train_images = None
train_labels = None
num_train_examples = 0
with open('train_data.csv', encoding="utf8") as file:
  reader = csv.reader(file, delimiter=',')
  num_train_examples = sum(1 for row in reader)

with open('train_data.csv', encoding="utf8") as file:
  reader = csv.reader(file, delimiter=',')
  train_images = np.zeros(shape = (num_train_examples - 1, EXAMPLE_LENGTH, glove_vector_dim))
  train_labels = np.zeros(shape = (num_train_examples - 1, 1))
  for idx, example in enumerate(reader):
    if idx == 0:
      continue
    _, subreddit, title, score, num_comments, timestamp = tuple(example)  
    title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()
    title_words_list = [x.lower() for x in title_words_list]
    for j in range(len(title_words_list)):
      curr_word = title_words_list[j]
      if curr_word in glove_vectors:
        train_images[idx-1][j] = glove_vectors[curr_word]
        train_labels[idx-1] = float(score)

# Read in dev/test data
test_images = None
test_labels = None
with open('dev_data.csv', encoding="utf8") as file:
  reader = csv.reader(file, delimiter=',')
  num_test_examples = sum(1 for row in reader)

with open('dev_data.csv', encoding="utf8") as file:
  reader = csv.reader(file, delimiter=',')
  test_images = np.zeros(shape = (num_test_examples - 1, EXAMPLE_LENGTH, glove_vector_dim))
  test_labels = np.zeros(shape = (num_test_examples - 1, 1))
  for idx, example in enumerate(reader):
    if idx == 0:
      continue
    _, subreddit, title, score, num_comments, timestamp = tuple(example)  
    title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()
    title_words_list = [x.lower() for x in title_words_list]
    for j in range(len(title_words_list)):
      curr_word = title_words_list[j]
      if curr_word in glove_vectors:
        test_images[idx-1][j] = glove_vectors[curr_word]
        test_labels[idx-1] = float(score)

# train_images, train_labels = generate_minibatch.export_main(input_size, batch_size)
train_images = torch.from_numpy(train_images)
train_labels = torch.from_numpy(train_labels)

# test_images, test_labels = generate_minibatch.export_main(input_size, batch_size)
test_images = torch.from_numpy(test_images)
test_labels = torch.from_numpy(test_labels)


# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True, 
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False, 
#                                           transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size, 
#                                           shuffle=False)

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        self.fc = nn.Linear(hidden_size*2, 1)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# Use mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
total_step = len(train_labels)
for epoch in range(num_epochs):
    for i in range(len(train_images)):
    # for i, (images, labels) in enumerate(train_loader):
        images = train_images[i].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
        labels = train_labels[i].type('torch.FloatTensor').to(device)
          
        # Forward pass
        outputs = model(images)
        # loss = criterion(outputs, labels)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(len(test_labels)):
    # for images, labels in test_loader:
        images = test_images[i].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
        labels = test_labels[i].type('torch.LongTensor').to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 100 test examples: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')