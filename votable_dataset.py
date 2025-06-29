from astropy.io.votable import parse
import os
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

"""
Here, we define a class for creating PyTorch tensors from votable files, 
suitable for creating the dataloaders in our main file
"""

class VotableDataset(): 

    """
    The class expects: 
    -> Directory where our votable files are located
    -> Class label (1 for Ha-emitters, 0 for non Ha-emitters)
    -> mode = {train, val, test}
    -> the ratio for splitting training dataset to train and validaton 
       (not used if mode = 'train')
    """

    def __init__(self, directory, label, mode, train_val_split = 0.0): 
        self.files = os.listdir(directory)
        self.directory = directory 
        self.label = label 
        self.mode = mode

        random.seed(42)

        split_id = int(len(self.files) * train_val_split)
        self.train_files = self.files[:split_id]
        self.val_files = self.files[split_id:]

        self.scaler = StandardScaler()
        
    def __len__(self):
        if self.mode == 'train' : 
            return len(self.train_files)
        elif self.mode == 'val':
            return len(self.val_files)
        return len(self.files)
    
    def __getitem__(self, idx): 

        if self.mode == 'train': 
            xml_file = self.train_files[idx]
        elif self.mode == 'val': 
            xml_file = self.val_files[idx]
        else: 
            xml_file = self.files[idx]

        """
        For ease, votable files are first converted to pandas dataframes, and then to pytorch tensors
        """
        votable_file = parse(os.path.join(self.directory, xml_file))
        table = votable_file.get_first_table().to_table().to_pandas()
        
        flux = table['flux'].values

        #normalised_flux = (flux - np.median(flux)) / np.std(flux)

        standardised_flux = self.scaler.fit_transform(flux.reshape(-1, 1)).flatten()


        flux_tensor = torch.tensor(standardised_flux, dtype = torch.float32)
        label_tensor = torch.tensor(self.label, dtype = torch.long)

        id = (xml_file.split('.')[0]).split(' ')[-1]

        return flux_tensor, label_tensor, id
