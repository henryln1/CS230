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

from reference_modified import BiRNN
from classification_model import ClassBiRNN

input_size = 50
hidden_size = 1024
num_layers = 2
BATCH_SIZE = 64

# Regression model hyperparams
r_num_classes = 1

# Classification model hyperparams
c_num_classes = 20

print("Loading GloVe vectors...")
glove = U.load_glove()

print("Reading in data...")
train_data, dev_data, test_data = U.get_data(glove = glove, device = device)

# print("Loading regression model...")
# class_model = BiRNN(glove, input_size, hidden_size, num_layers, r_num_classes).to(device)

print("Loading classification model...")
regression_model = ClassBiRNN(glove, input_size, hidden_size, num_layers, c_num_classes).to(device)

# Test regression model
# output_path = C.filenames['bi_rnn']
# print("Testing regression model...")
# U.test_model(model, output_path, test_data, BATCH_SIZE, outputs = 'regression_results.txt')

# Test classification model
output_path = C.filenames['class_rnn']
print("Testing classification model...")
U.test_model(class_model, output_path, test_data, BATCH_SIZE, classification = True, outputs = 'classification_results.txt')