# https://github.com/yunjey/pytorch-tutorial
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import generate_minibatch
import time
import csv
import numpy as np
import re
import random
from sklearn.metrics import mean_squared_error

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
batch_size = 512
num_epochs = 4
learning_rate = 0.0001
dev_batch_size = 512

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
dev_images = None
dev_labels = None
with open('dev_data.csv', encoding="utf8") as file:
  reader = csv.reader(file, delimiter=',')
  num_dev_examples = sum(1 for row in reader)

with open('dev_data.csv', encoding="utf8") as file:
  reader = csv.reader(file, delimiter=',')
  dev_images = np.zeros(shape = (num_dev_examples - 1, EXAMPLE_LENGTH, glove_vector_dim))
  dev_labels = np.zeros(shape = (num_dev_examples - 1, 1))
  for idx, example in enumerate(reader):
    if idx == 0:
      continue
    _, subreddit, title, score, num_comments, timestamp = tuple(example)  
    title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()
    title_words_list = [x.lower() for x in title_words_list]
    for j in range(len(title_words_list)):
      curr_word = title_words_list[j]
      if curr_word in glove_vectors:
        dev_images[idx-1][j] = glove_vectors[curr_word]
        dev_labels[idx-1] = float(score)

# train_images, train_labels = generate_minibatch.export_main(input_size, batch_size)
train_images = torch.from_numpy(train_images)
train_labels = torch.from_numpy(train_labels)

# dev_images, dev_labels = generate_minibatch.export_main(input_size, batch_size)
dev_images = torch.from_numpy(dev_images)
dev_labels = torch.from_numpy(dev_labels)


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
number_params = count_parameters(model)
print("Number of trainable parameters: ", number_params)

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# Use mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# checks weights are updating part 1
# old_state_dict = {}
# for key in model.state_dict():
#     old_state_dict[key] = model.state_dict()[key].clone()


dev_error_history = 'dev_model_1_error_history.txt'
train_error_history = 'train_model_1_error_history.txt'

train_f = open(train_error_history, "a+")
dev_f = open(dev_error_history, "a+")

# Train the model
total_step = len(train_labels)
for epoch in range(num_epochs):
    start_time = time.time()
    for i in range(len(train_images)):
    # for i in range(100): #training on small set
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
            # checks weights are updating part 2
            # new_state_dict = {}
            # for key in model.state_dict():
            #     new_state_dict[key] = model.state_dict()[key].clone()
            # for key in old_state_dict:
            #     if not (old_state_dict[key] == new_state_dict[key]).all():
            #         print('Diff in {}'.format(key)

        if (i+1) % 1000 == 0:
            print('Evaluating on last 500 examples from training set.')
            with torch.no_grad():
                correct = 0
                total = 0
                predicted_list = []
                actual_label_list = []
                for j in range(i, i - 500, -1):
                    images = dev_images[j].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
                    labels = dev_labels[j].type('torch.FloatTensor').to(device)
                    outputs = model(images)
                    predicted, _ = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    actual_label_list.append(dev_labels[j])
                    predicted_list.append(predicted.item())
                print('Train Accuracy of the model on the 500 training examples: {} %'.format(100 * correct / total))               
                print('Mean Squared Error: ', mean_squared_error(predicted_list, actual_label_list))
                train_f.write("Predicted: " + str(predicted_list) + '\n')
                train_f.write("Actual: " + str(actual_label_list) + '\n')
                train_f.write("MSE: " + str(mean_squared_error(predicted_list, actual_label_list)) + '\n')

            print('Next, evaluating on 500 examples from dev set.')
            with torch.no_grad():
                correct = 0
                total = 0
                random_start = random.randint(1, len(dev_labels) - 1000)
                predicted_list = []
                actual_label_list = []
                for j in range(random_start, random_start + 500):
                    images = dev_images[j].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
                    labels = dev_labels[j].type('torch.FloatTensor').to(device)
                    outputs = model(images)
                    predicted, _ = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    actual_label_list.append(dev_labels[j])
                    predicted_list.append(predicted.item())
                print('Dev Accuracy of the model on the 500 dev examples: {} %'.format(100 * correct / total)) 
                print('Mean Squared Error: ', mean_squared_error(predicted_list, actual_label_list))
                dev_f.write("Predicted: " + str(predicted_list) + '\n')
                dev_f.write("Actual: " + str(actual_label_list) + '\n')
                dev_f.write("MSE: " + str(mean_squared_error(predicted_list, actual_label_list)) + '\n')

    time_elapsed = time.time() - start_time
    print("Epoch: " , str(epoch + 1), " took ", str(time_elapsed), " seconds.")


dev_output_file = 'dev_model_outputs_1.txt'
train_output_file = 'train_model_outputs_1.txt'
# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    predicted_list = []
    actual_label_list = []
    for i in range(len(dev_labels)):
    # for images, labels in test_loader:
        images = dev_images[i].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
        labels = dev_labels[i].type('torch.FloatTensor').to(device)
        outputs = model(images)
        predicted, _ = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        actual_label_list.append(dev_labels[i])
        predicted_list.append(predicted.item())
    print('Dev Accuracy of the model on the full dev set: {} %'.format(100 * correct / total)) 
    print('Mean Squared Error: ', mean_squared_error(predicted_list, actual_label_list))

print("Predicted: ", predicted_list)
print("Actual: ", actual_label_list)
f = open(dev_output_file, "a+")
f.write('Predicted: ' + str(predicted_list) + '\n')
f.write('Actual: ' + str(actual_label_list) + '\n')



with torch.no_grad():
    correct = 0
    total = 0
    predicted_list = []
    actual_label_list = []
    for i in range(len(train_labels)):
    # for images, labels in test_loader:
        images = train_images[i].reshape(-1, sequence_length, input_size).type('torch.FloatTensor').to(device)
        labels = train_labels[i].type('torch.FloatTensor').to(device)
        outputs = model(images)
        predicted, _ = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        actual_label_list.append(train_labels[i])
        predicted_list.append(predicted.item())
    print('Dev Accuracy of the model on the full dev set: {} %'.format(100 * correct / total)) 
    print('Mean Squared Error: ', mean_squared_error(predicted_list, actual_label_list))
    
print("Predicted: ", predicted_list)
print("Actual: ", actual_label_list)
f = open(train_output_file, "a+")
f.write('Predicted: ' + str(predicted_list) + '\n')
f.write('Actual: ' + str(actual_label_list) + '\n')

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')