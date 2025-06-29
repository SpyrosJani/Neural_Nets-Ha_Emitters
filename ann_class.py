"""
Here we define the architecture of the ANN model 
"""

import torch.nn.functional as F
from torch import nn
import torch

class ANN(nn.Module): 
    """
    The __init__() function defines the layers 
    and the number of neurons used per layer 
    as well as dropout and other regularisation techniques
    """
    def __init__(self, input_size, hidden_sizes, dropouts, num_classes = 2):
        
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        # 1st dropout
        self.dropout1 = nn.Dropout(dropouts[0])

        # Hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # 2nd dropout 
        self.dropout2 = nn.Dropout(dropouts[1])

        # Output layer, which maps to the number of classes we want for our classification task
        self.out = nn.Linear(hidden_sizes[1], num_classes)
    

    """
    This function defineds how the data are passing through the model
    """
    def forward(self, x): 

        # Firstly, flattern the input data, in order to have the shape [batch_size * (height*width)]
        x = x.flatten(1)

        # Pass data through the first layer 
        x = self.fc1(x)
        x = F.relu(x)

        # Apply 1st dropout
        x = self.dropout1(x)

        # Do the same for the next layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Pass data throught output layer and map them in the output classes
        output = self.out(x)

        return output