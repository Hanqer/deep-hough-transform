import numpy as np 
import os
from os.path import join, split, isdir, isfile, abspath
import torch
from PIL import Image
import random
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class SemanLineDatasetTest(Dataset):

    def __init__(self, root_dir, label_file, transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.image_path = [join(root_dir, i) for i in lines]
        self.transform = transform
        self.t_transform = t_transform
        
    def __getitem__(self, item):
        assert isfile(self.image_path[item]), self.image_path[item]
        image = Image.open(self.image_path[item]).convert('RGB')
        w, h = image.size
        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.image_path[item], [h, w]


    def __len__(self):
        return len(self.image_path)

def get_loader(root_dir, label_file, batch_size, img_size=0, num_thread=4, pin=True, test=False, split='train'):
    if test is False:
        print("Not implemented the dataloader for training")
        exit()
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

        
