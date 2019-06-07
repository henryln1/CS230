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

BATCH_SIZE = 256

print("Loading GloVe vectors...")
glove = U.load_glove()

print("Reading in data...")
train_data, dev_data, test_data = U.get_data(glove = glove, device = device)

print("Loading model...")
model = BiRNN(glove, input_size, hidden_size, num_layers, num_classes).to(device)

# Test regression model
output_path = C.filenames['bi_rnn']
print("Testing regression model...")
test_model(model, output_path, test_data, BATCH_SIZE)

# Test classification model
output_path = C.filenames['class_rnn']
print("Testing classification model...")
U.test_model(model, output_path, test_data, BATCH_SIZE, classification = True)