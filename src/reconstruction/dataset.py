import os

import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor

'''
Generates a basic face dataset with feature labels.
'''
class FaceDataset(Dataset):
    
    def __init__(self, good_data, transforms=None):
        # Load the features.
        self.img_names = good_data.index.values
        self.features = good_data.values
        self.transform = transforms
        
    def __getitem__(self, index):
        # Transform images by downsampling and then converting to tensor.
        #img = self.images[index]
        datadir = '/home/declan/Data/Faces/'
        img = Image.open(os.path.join(datadir, 'CelebA-HQ-img', self.img_names[index]))
        features = self.features[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, features
        
    def __len__(self):
        return self.features.shape[0]

        
        