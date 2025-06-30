"""
Here we define various pipelines and tools we use in the training, mainly: 
-> The training pipeline per epoch 
-> The validation pipeline per epoch
-> The testing (inference) routine 
-> The EarlyStopper class 
"""

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt

def train(dataloader, model, criterion, optimizer, scheduler, device): 

    running_loss = 0

    correct_labels = 0
    total_labels = 0

    for X, y in dataloader: 

        optimizer.zero_grad()

        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = criterion(pred, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        probabilites, predicted_labels = torch.max(pred, 1)

        total_labels += y.size(0)
        correct_labels += (predicted_labels == y).sum().item()
    
    scheduler.step()

    print(f"Learning rate for this epoch is: {scheduler.get_lr()}")

    train_accuracy = 100 * correct_labels/total_labels
    train_loss = running_loss / len(dataloader)

    print(f"Training accuracy: {train_accuracy}%, Training Loss: {train_loss}")

    return train_accuracy, train_loss

def validate(dataloader, model, criterion, device): 
    
    running_loss = 0

    correct_labels = 0 
    total_labels = 0 

    with torch.no_grad(): 

        for X, y in dataloader: 

            X, y = X.to(device), y.to(device)

            pred = model(X)

            loss = criterion(pred, y)

            running_loss += loss.item()
            probabilities, predicted_labels = torch.max(pred, 1)

            total_labels += y.size(0)
            correct_labels += (predicted_labels == y).sum().item()
    
    val_accuracy = 100 * correct_labels/total_labels
    val_loss = running_loss / len(dataloader)

    print(f"Validation accuracy: {val_accuracy}%, Validation Loss: {val_loss}")

    return val_accuracy, val_loss

def test(dataloader, model, device, criterion):

    # Model must be set to evaluation mode 
    model.eval()

    running_loss = 0.0

    predictions = []
    all_labels = []

    with torch.no_grad():
        for X, y, id in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            pred = model(X)
            
            # Calculate loss
            loss = criterion(pred, y)
            running_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(pred.data, 1)

            #This commented segment is used for finding 
            #which GAIA spectra are misclassified. 
            '''
            if (predicted != y): 
                print("Misclassification on object with id: ", id)
                plt.plot(np.arange(336, 1022, 2), X.cpu().numpy()[0])
                plt.title("Spectrum of undetected Hα emitter")
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Standardized Flux Value')
                plt.show()
            '''
            # Store predictions and labels
            predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    test_accuracy = 100 * np.sum(np.array(predictions) == np.array(all_labels)) / len(all_labels)
    test_loss = running_loss / len(dataloader)

    print(f"Testing accuracy: {test_accuracy:.2f}%, Testing Loss: {test_loss:.4f}")

    # Calculate precision, recall, f1-score
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Caclulate and display confusion matrix
    conf_matrix = confusion_matrix(all_labels, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['non-Hα\nemitter', 'Hα\nemitter'])
    plt.figure(figsize = (10,8))
    disp.plot(cmap = 'Blues', colorbar = False, text_kw={'fontsize': 50, 'weight': 'bold'})
    ax = disp.ax_
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=50, ha = 'center', rotation = 0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=50, va = 'center')
    ax.set_position([0.25, 0.25, 0.6, 0.6])  # [left, bottom, width, height]

    plt.title('Confusion Matrix for CNN', fontsize=54, pad=40, weight='bold')
    plt.xlabel('Predicted Label', fontsize=50, labelpad = 30, style = 'italic')
    plt.ylabel('True Label', fontsize=50, labelpad = 30, style = 'italic')
    plt.show()
    #plt.savefig('FCN_Confusion_Matrix.png')
    #plt.close()

    # Wilson score interval
    accuracy_frac = np.mean(np.array(predictions) == np.array(all_labels))
    ci_low, ci_up = proportion_confint(count=int(accuracy_frac * len(all_labels)), nobs=len(all_labels), alpha=0.05, method='wilson')
    print(f"95% Wilson CI for Accuracy: [{ci_low*100:.2f}%, {ci_up*100:.2f}%]")


    return test_accuracy, test_loss


class EarlyStopper: 
    
    def __init__(self, patience):

        self.patience = patience
        self.counter = 0 
        self.min_loss = float('inf')

    def early_stop(self, validation_loss): 

        """
        Use validation loss as metric 
        If after for a consecutive specified number of epochs 
        the validation loss is not reduced, terminate the training 
        in order to avoid overfitting 
        """
        if validation_loss <= self.min_loss: 

            self.min_loss = validation_loss
            self.counter = 0

        else: 

            self.counter += 1
            if self.counter > self.patience: 
                return True 
        
        return False





        

