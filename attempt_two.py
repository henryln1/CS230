import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import generate_minibatch
import numpy as np


lstm_input_size = 50
num_train = 128
output_dim = 1
h1 = 128
num_layers = 2
learning_rate = 0.001
num_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        # if not self.h:
        # 	self.h = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        # 	self.c = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)

        # lstm_out, _ = self.lstm(input)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)



loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)



######
# Load model
######
glove_vector_dimensions = 50
data_file_location = './preprocessing_code/RS_2018-09_AskReddit_submissions_commas_removed.csv'
glove_vector_location = './preprocessing_code/glove.6B/glove.6B.' + str(glove_vector_dimensions) + 'd.txt'
glove_vectors_dict = generate_minibatch.build_glove_dict(glove_vector_location)
X_train, y_train = generate_minibatch.build_minibatch(num_train, glove_vectors_dict, data_file_location)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Clear stored gradient
    model.zero_grad()
    
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()
    
    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()










