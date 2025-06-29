import torch.nn.functional as F 
from torch import nn 


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel):
        super(ResidualBlock, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.0)
        )

        self.b2 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm1d(channels)
        )
    
    def forward(self, x): 

        residual = x 
        out = self.b1(x)
        out = self.b2(out)

        out += residual 
        output = F.relu(out)

        return output

class CNN(nn.Module):
    def __init__(self, input_size, num_classes = 2):
        
        super(CNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # 1st convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.MaxPool1d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*(343//2), 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, self.num_classes)
        )
    

    """
    This function defineds how the data are passing through the model
    """
    def forward(self, x): 

        #print(x.shape)
        x = x.unsqueeze(1)

        x = self.conv1(x)

        x = self.conv2(x)

        #x = self.conv3(x)

        #x = self.conv4(x)

        output = self.classifier_head(x)
        #print(output.shape)


        return output


class ResCNN(nn.Module):
    def __init__(self, input_size, num_classes = 2):
        
        super(ResCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # 1st convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv2 = ResidualBlock(
            channels=32, 
            kernel = 7
        )

        self.conv3 = ResidualBlock(
            channels = 32, 
            kernel = 7
        )
        
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*(343//2), 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, self.num_classes)
        )
    

    """
    This function defineds how the data are passing through the model
    """
    def forward(self, x): 

        #print(x.shape)
        x = x.unsqueeze(1)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        output = self.classifier_head(x)
        #print(output.shape)


        return output