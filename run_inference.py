from cnn_class import CNN
from ann_class import ANN
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os  
from utils import test
from votable_dataset import VotableDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 343
batch_size = 1
test_path = './sampled_spectra/test'


test_detected = VotableDataset(directory = os.path.join(test_path, 'Ha_detected'),
                                   label = 1, mode = 'test')
test_noDetected = VotableDataset(directory = os.path.join(test_path, 'Ha_noDetected'),
                                label = 0, mode = 'test')
test_dataset = ConcatDataset([test_detected, test_noDetected])
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


best_model = CNN(input_size=input_size, num_classes=2)
#best_model = ANN(input_size = input_size, hidden_sizes = [128, 32], dropouts = [0, 0], num_classes = 2)
best_model.load_state_dict(torch.load('./sampled_spectra/checkpoints/best_model_cnn.pth'))
best_model.to(device)
test_acc, test_loss = test(test_loader, best_model, device, criterion = nn.CrossEntropyLoss())
