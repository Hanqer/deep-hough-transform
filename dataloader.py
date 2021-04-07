import numpy as np 
import os
from os.path import join, split, isdir, isfile, abspath
import torch
from PIL import Image
import random
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



class SemanLineDataset(Dataset):

    def __init__(self, root_dir, label_file, split='train', transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.image_path = [join(root_dir, i+".jpg") for i in lines]
        self.data_path = [join(root_dir, i+".npy") for i in lines]
        self.split = split
        self.transform = transform
        self.t_transform = t_transform
    
    def __getitem__(self, item):

        assert isfile(self.image_path[item]), self.image_path[item]
        image = Image.open(self.image_path[item]).convert('RGB')

        data = np.load(self.data_path[item], allow_pickle=True).item()
        hough_space_label8 = data["hough_space_label8"].astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)
            
        hough_space_label8 = torch.from_numpy(hough_space_label8).unsqueeze(0)
        if self.split == 'val':
            gt_coords = data["coords"]
            return image, hough_space_label8, gt_coords, self.image_path[item].split('/')[-1]
        elif self.split == 'train':
            return image, hough_space_label8, self.image_path[item].split('/')[-1]

    def __len__(self):
        return len(self.image_path)

class SemanLineDatasetTest(Dataset):

    def __init__(self, root_dir, label_file, transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.image_path = [join(root_dir, i+".jpg") for i in lines]
        self.transform = transform
        self.t_transform = t_transform
        
    def __getitem__(self, item):

        assert isfile(self.image_path[item]), self.image_path[item]
        image = Image.open(self.image_path[item]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.image_path[item].split('/')[-1]


    def __len__(self):
        return len(self.image_path)

def get_loader(root_dir, label_file, batch_size, img_size=0, num_thread=4, pin=True, test=False, split='train'):
    if test is False:
        transform = transforms.Compose([
        # transforms.Resize((400, 400)),#   Not used for current version.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = SemanLineDataset(root_dir, label_file, transform=transform, t_transform=None, split=split)
    else:
        transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = SemanLineDatasetTest(root_dir, label_file, transform=transform, t_transform=None)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                    pin_memory=pin)
    return data_loader

        
