"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import generate_minibatch

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 50      # rnn input size
LR = 0.02           # learning rate


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, words, glove dimensionality)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        # outs = []    # save all predictions
        # for time_step in range(r_out.size(1)):    # calculate output for each time step
        #     print("R OUT SHAPE: ", r_out.shape)
        #     outs.append(self.out(r_out[:, time_step, :]))
        # return torch.stack(outs, dim=1), h_state
        return self.out(r_out[:, -1, :]), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state
        
        # or even simpler, since nn.Linear can accept inputs of any dimension 
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

######
# Load model
######
glove_vector_dimensions = 50
data_file_location = './preprocessing_code/RS_2018-09_AskReddit_submissions_commas_removed.csv'
glove_vector_location = './preprocessing_code/glove.6B/glove.6B.' + str(glove_vector_dimensions) + 'd.txt'
glove_vectors_dict = generate_minibatch.build_glove_dict(glove_vector_location)
X_train, y_train = generate_minibatch.build_minibatch(100, glove_vectors_dict, data_file_location)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_train, y_train = X_train.type(torch.FloatTensor), y_train.type(torch.FloatTensor)
print(X_train.shape)
print(y_train.shape)




for step in range(100):

    # x = torch.from_numpy(X_train)    # shape (batch, time_step, input_size)
    # y = torch.from_numpy(y_train)

    x = X_train
    y = y_train

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # calculate loss
    print("Predicted: ", prediction[:50])
    print("Actual: ", y[:50])
    print("Loss: ", loss)
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

