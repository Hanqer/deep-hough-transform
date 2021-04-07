import numpy as np
import cv2
from PIL import Image
import argparse
import os, sys
from os.path import join, split, splitext, abspath, isfile
sys.path.insert(0, abspath(".."))
sys.path.insert(0, abspath("."))
from utils import Line, LineAnnotation, line2hough
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

parser = argparse.ArgumentParser(description="Prepare semantic line data format.")
parser.add_argument('--root', type=str, required=True, help='the data root dir.')
parser.add_argument('--label', type=str, required=True, help='the label root dir.')
parser.add_argument('--num_directions', type=int, default=12, help='the division of semicircular angle')
parser.add_argument('--list', type=str, required=True, help='list file')
parser.add_argument('--save-dir', type=str, required=True, help='save-dir')
parser.add_argument('--prefix', type=str, default="", help="Prefix in list file")
parser.add_argument('--fixsize', type=int, default=None, help='fix resize of images and annotations')
parser.add_argument('--numangle', type=int, default=80, required=True)
parser.add_argument('--numrho', type=int, default=80, required=True)
args = parser.parse_args()

label_path = abspath(args.label)
image_dir = abspath(args.root)
save_dir = abspath(args.save_dir)
os.makedirs(save_dir, exist_ok=True)
def nearest8(x):
    return int(np.round(x/8)*8)

def vis_anno(image, annotation):
    mask = annotation.oriental_mask()
    mask_sum = mask.sum(axis=0).astype(bool)
    image_cp = image.copy()
    image_cp[mask_sum, ...] = [0, 255, 0]
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[mask_sum] = 1
    return image_cp, mask


labels_files = [i for i in os.listdir(label_path) if i.endswith(".txt")]

num_samples = len(labels_files)

filelist = open(args.list, "w")
stastic = np.zeros(10)

for idx, label_file in enumerate(labels_files):
    filename, _ = splitext(label_file)
    print("Processing %s [%d/%d]..." % (filename, idx+1, len(labels_files)))
    if isfile(join(image_dir, filename+".jpg")):
        im = cv2.imread(join(image_dir, filename+".jpg"))
        im = cv2.resize(im, (args.fixsize, args.fixsize))
    else:
        print("Warning: image %s doesnt exist!" % join(image_dir, filename+".jpg"))
        continue
    for argument in range(2):
        if argument == 0:
            H, W, _ = im.shape
            lines = []
            with open(join(label_path, label_file)) as f:
                data = f.readlines()
                nums = len(data)
                stastic[nums] += 1
                for line in data:
                    data1 = line.strip().split(',')
                    if len(data1) <= 4:
                        continue
                    data1 = [int(float(x)) for x in data1]
                    if data1[1]==data1[3] and data1[0]==data1[2]:
                        continue
                    line = Line([data1[1], data1[0], data1[3], data1[2]])
                    lines.append(line)
            
            annotation = LineAnnotation(size=[H, W], divisions=args.num_directions, lines=lines)
        else:
            im = cv2.flip(im, 1)
            filename = filename + '_flip'
            H, W, _ = im.shape
            lines = []
            with open(join(label_path, label_file)) as f:
                data = f.readlines()
                for line in data:
                    data1 = line.strip().split(',')
                    if len(data1) <= 4:
                        continue 
                    data1 = [int(float(x)) for x in data1]
                    if data1[1]==data1[3] and data1[0]==data1[2]:
                        continue
                    line = Line([data1[1], W-1-data1[0], data1[3], W-1-data1[2]])
                    lines.append(line)
            
            annotation = LineAnnotation(size=[H, W], divisions=args.num_directions, lines=lines)

        # resize image and annotations
        if args.fixsize is not None:
            newH = nearest8(args.fixsize)
            newW = nearest8(args.fixsize)
        else:
            newH = nearest8(H)
            newW = nearest8(W)

        im = cv2.resize(im, (newW, newH))
        annotation.resize(size=[newH, newW])
        vis, mask = vis_anno(im, annotation)

        hough_space_label = np.zeros((args.numangle, args.numrho))
        for l in annotation.lines:
            theta, r = line2hough(l, numAngle=args.numangle, numRho=args.numrho, size=(newH, newW))
            hough_space_label[theta, r] += 1

        hough_space_label = cv2.GaussianBlur(hough_space_label, (5,5), 0)

        if hough_space_label.max() > 0:
            hough_space_label = hough_space_label / hough_space_label.max()

        gt_coords = []
        for l in annotation.lines:
            gt_coords.append(l.coord)
        gt_coords = np.array(gt_coords)
        data = dict({
            "hough_space_label8": hough_space_label,
            "coords": gt_coords
        })

        save_name = os.path.join(save_dir, filename)

        np.save(save_name, data)
        cv2.imwrite(save_name + '.jpg', im)
        cv2.imwrite(save_name + '_p_label.jpg', hough_space_label*255)
        cv2.imwrite(save_name + '_vis.jpg', vis)
        cv2.imwrite(save_name + '_mask.jpg', mask*255)
filelist.close()
# print(stastic)

