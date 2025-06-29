"""
This script does the following: 
-> Set the parameters of the training 
-> Create the dataloaders 
-> Execute training - validation
-> Extract corresponding plots
"""

from ann_class import ANN
from cnn_class import CNN, ResCNN
from utils import train, validate, test, EarlyStopper
from votable_dataset import VotableDataset

import os 
import matplotlib.pyplot as plt
import torch 
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets
from torchsummary import summary
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

# Set the relevant parameters
lr = 0.0005
num_epochs = 15
batch_size = 1

"""
-> Images expect input size of 128*128
-> Votable files expect 343
"""
filetype = "votable"
input_size = 256*256 if filetype == 'png' else 343
val_best_accuracy = 0.0

# Set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Defines image transformations 
# TO-DO: 
#       -> calculate mean and deviation of dataset in separate script
#       -> see possible methods for augmentation (flipping, rotating, darkening, etc.)
transform = v2.Compose ([
    v2.ToImage(),
    v2.Grayscale(num_output_channels = 1),
    v2.Resize((256, 256)),
    v2.ToDtype(torch.float32, scale = True),
    #v2.Normalize(mean = [0], std = [1])
])

# Create firstly the train and the test datasets 
train_path = './sampled_spectra/train'
test_path = './sampled_spectra/test'

train_subset = []
val_subset = []
test_subset = []

"""
Depending on whether we use images or votable files as our dataset, 
the dataloaders must be constructed in a different way
"""
if filetype == "png":

    """
    Augment image data artificially
    """
    train_dataset = datasets.ImageFolder(root = train_path, transform = transform)
    test_dataset = datasets.ImageFolder(root = test_path, transform = transform)

    # Split training and validation into balanced sets
    detected_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 0]  
    noDetected_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 1]  

    train_detected, val_detected = train_test_split(detected_indices, train_size = 0.75, random_state = 42)
    train_noDetected, val_noDetected = train_test_split(noDetected_indices, train_size = 0.75, random_state = 42)

    train_indices = train_detected + train_noDetected
    val_indices = val_detected + val_noDetected
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)


elif filetype == "votable": 
    """
    Define in the arguments the splitting ratio for train-val split
    """
    train_detected = VotableDataset(directory = os.path.join(train_path, 'Ha_detected'),
                                    label = 1, mode = 'train', train_val_split = 0.8)
    train_noDetected = VotableDataset(directory = os.path.join(train_path, 'Ha_noDetected'),
                                    label = 0, mode = 'train', train_val_split = 0.8)
    train_subset = ConcatDataset([train_detected, train_noDetected])


    val_detected = VotableDataset(directory = os.path.join(train_path, 'Ha_detected'),
                                    label = 1, mode = 'val', train_val_split=0.8)
    val_noDetected = VotableDataset(directory = os.path.join(train_path, 'Ha_noDetected'),
                                    label = 0, mode = 'val', train_val_split=0.8)
    val_subset = ConcatDataset([val_detected, val_noDetected])



    test_detected = VotableDataset(directory = os.path.join(test_path, 'Ha_detected'),
                                   label = 1, mode = 'test')
    test_noDetected = VotableDataset(directory = os.path.join(test_path, 'Ha_noDetected'),
                                   label = 0, mode = 'test')
    test_dataset = ConcatDataset([test_detected, test_noDetected])

print(f"Training dataset length: {len(train_subset)}")
print(f"Validation dataset length: {len(val_subset)}")
print(f"Test dataset length: {len(test_dataset)}")


# Create the PyTorch Dataloaders
train_loader = DataLoader(train_subset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_subset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


# Instantiate model
#model = ANN(input_size = input_size, hidden_sizes = [128, 32], dropouts = [0, 0], num_classes = 2)
model = CNN(input_size=input_size, num_classes=2)
print(summary(model, (343,)))
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)
early_stopper = EarlyStopper(patience = 5)


train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []
for epoch in range(num_epochs): 

    print(f"Training in epoch: {epoch}")
    model.train()
    train_accuracy, train_loss = train(train_loader, model, criterion, optimizer, scheduler, device)

    print(f"Validating in epoch: {epoch}")
    model.eval()
    val_accuracy, val_loss = validate(test_loader, model, criterion, device)

    if (val_accuracy > val_best_accuracy): 

        val_best_accuracy = val_accuracy
        torch.save(model.state_dict(), "./sampled_spectra/checkpoints/best_model_cnn.pth")
    
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (early_stopper.early_stop(val_loss)): 
        print("Early Stopping has been activated!")
        break 

plt.figure(figsize = (10, 5))
plt.plot(range(len(train_losses)), train_losses, label = 'Train Loss')
plt.plot(range(len(val_losses)), val_losses, label = 'Validation Loss')
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.title('Loss per epoch')
plt.savefig('./sampled_spectra/Loss_v_Epochs_cnn.png')

plt.figure(figsize = (10, 5))
plt.plot(range(len(train_accuracies)), train_accuracies, label = 'Train Accuracy')
plt.plot(range(len(val_accuracies)), val_accuracies, label = 'Validation Accuracy')
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy per epoch')
plt.savefig('./sampled_spectra/Accuracy_v_Epochs_cnn.png')

#best_model = ANN(input_size = input_size, hidden_sizes = [128, 32], dropouts = [0, 0], num_classes = 2)
best_model = CNN(input_size=input_size, num_classes=2)
best_model.load_state_dict(torch.load('./sampled_spectra/checkpoints/best_model_cnn.pth'))
best_model.to(device)
test_acc, test_loss = test(test_loader, best_model, device, criterion = nn.CrossEntropyLoss())