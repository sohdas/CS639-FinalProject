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
    
    def __init__(self, transforms=None):
        # Load the features.
        attribute_file = 'CelebAMask-HQ-attribute-anno.txt'
        datadir = '/home/declan/Data/Faces/'
        img_dir = os.path.join(datadir, 'CelebA-HQ-img')
        attr_dir = os.path.join(datadir, 'CelebAMask-HQ-attribute-anno.txt')
        attributes = pd.read_csv(attr_dir, delimiter=' ')
        good_data = attributes[['Bald', 'Smiling', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necktie', 'Wearing_Necklace', 'Gray_Hair', 'Eyeglasses']]
        good_data = good_data.sample(frac=1) ##### CHANGE THIS LATER ######
        good_data = good_data.clip(lower=0) # Set -1 to 0.
        good_data = good_data[(good_data.T != 0).any()] # drop any rows with only zeros.
        self.img_names = good_data.index.values
        #self.images = [Image.open(os.path.join(datadir, 'CelebA-HQ-img', img)) for img in self.img_names]
        #for img_name in img_names:
        #    self.images.append(Image.open(os.path.join(datadir, 'CelebA-HQ-img', img_name)))
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

        
        