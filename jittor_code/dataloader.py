import numpy as np 
import os
from os.path import join, split, isdir, isfile, abspath
from PIL import Image
import random
import collections

import jittor.transform as transforms
import jittor as jt
from jittor.dataset import Dataset

class SemanLineDatasetTest(Dataset):

    def __init__(self, root_dir, label_file, transform=None, t_transform=None):
        super().__init__()
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.image_path = [join(root_dir, i+".jpg") for i in lines]
        self.transform = transform
        self.t_transform = t_transform
        self.set_attrs(total_len=len(self.image_path), keep_numpy_array=True)
        
    def __getitem__(self, item):

        assert isfile(self.image_path[item]), self.image_path[item]
        image = Image.open(self.image_path[item]).convert('RGB')
        w, h = image.size
        if self.transform is not None:
            image = self.transform(image)
        return image, self.image_path[item].split('/')[-1], (h, w)

    def collate_batch(self, batch):
        images, names, sizes = list(zip(*batch))
        images = jt.stack([jt.array(image) for image in images])
        return images, names, sizes

def get_loader(root_dir, label_file, batch_size, img_size=0, num_thread=4, pin=True, test=False, split='train'):
    if test is False:
        raise NotImplementedError
    else:
        transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = SemanLineDatasetTest(root_dir, label_file, transform=transform, t_transform=None)
    if test is False:
        raise NotImplementedError
    else:
        dataset.set_attrs(batch_size=batch_size, shuffle=False)
        print('Get dataset success.')
    return dataset


