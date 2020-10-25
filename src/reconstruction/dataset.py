import os

from PIL import Image
import pandas as pd
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor

'''
Generates a basic face dataset with feature labels.
'''
class FaceDataset(Dataset):
    
    def __init__(self, transforms=None):
        # Load the features.
        attribute_file = 'CelebAMask-HQ-attribute-anno.txt'
        datadir = '/home/declan/Data/Faces/'
        img_dir = os.path.join(datadir, 'CelebA-HQ-img')
        attr_dir = os.path.join(datadir, 'CelebAMask-HQ-attribute-anno.txt')
        attributes = pd.read_csv(attr_dir, delimiter=' ')
        good_data = attributes[['Bald', 'Smiling', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necktie', 'Wearing_Necklace', 'Gray_Hair', 'Eyeglasses']]
        good_data = good_data.sample(frac=.01) ##### CHANGE THIS LATER ######
        good_data = good_data.clip(lower=0) # Set -1 to 0.
        good_data = good_data[(good_data.T != 0).any()] # drop any rows with only zeros.
        img_names = good_data.index.values
        self.images = [Image.open(os.path.join(datadir, 'CelebA-HQ-img', img)) for img in img_names]
        self.features = good_data.values
        self.transform = transforms
        
    def __getitem__(self, index):
        # Transform images by downsampling and then converting to tensor.
        img = self.images[index]
        features = self.features[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, features
        
    def __len__(self):
        return len(self.images)

        
        