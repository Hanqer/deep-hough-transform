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
parser.add_argument('--save-dir', type=str, required=True, help='save-dir')
parser.add_argument('--fixsize', type=int, default=None, help='fix resize of images and annotations')
parser.add_argument('--numangle', type=int, default=100)
parser.add_argument('--numrho', type=int, default=100)
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

def check(y, x, H, W):
    x = max(0, x)
    y = max(0, y)
    x = min(x, W-1)
    y = min(y, H-1)
    return y, x

labels_files = [i for i in os.listdir(label_path) if i.endswith(".txt")]

num_samples = len(labels_files)

stastic = np.zeros(20)
total_nums = 0
angle_stastic = np.zeros(180)
total_lines = 0


for idx, label_file in enumerate(labels_files):
    filename, _ = splitext(label_file)
    print("Processing %s [%d/%d]..." % (filename, idx+1, len(labels_files)))
    if isfile(join(image_dir, filename+".jpg")):
        im = cv2.imread(join(image_dir, filename+".jpg"))
        H, W = im.shape[0], im.shape[1]
        scale_H, scale_W = args.fixsize / H, args.fixsize / W
        im = cv2.resize(im, (args.fixsize, args.fixsize))
    else:
        print("Warning: image %s doesnt exist!" % join(image_dir, filename+".jpg"))
        continue
    for argument in range(2):
        if argument == 0:
            lines = []
            with open(join(label_path, label_file)) as f:
                data = f.readlines()[0].split(' ')
                nums = int(data[0])
                stastic[nums] += 1
                total_nums += nums
                
                if int(nums) == 0:
                    print("Warning: image has no semantic line : %s" % (filename))

                for i in range(nums):
                    y1, x1 = check(int(data[i*4+2]), int(data[i*4+1]), H, W)
                    y2, x2 = check(int(data[i*4+4]), int(data[i*4+3]), H, W)
                    line = Line([y1, x1, y2, x2])
                    angle = line.angle()
                    angle_stastic[int((angle / np.pi + 0.5) * 180)] += 1
                    total_lines += 1
                    line.rescale(scale_H, scale_W)
                    lines.append(line)
            
            annotation = LineAnnotation(size=[args.fixsize, args.fixsize], lines=lines)
        else:
            im = cv2.flip(im, 1)
            filename = filename + '_flip'
            lines = []
            with open(join(label_path, label_file)) as f:
                data = f.readlines()[0].split(' ')
                for i in range(int(data[0])):
                    y1, x1 = check(int(data[i*4+2]), W-1-int(data[i*4+1]), H, W)
                    y2, x2 = check(int(data[i*4+4]), W-1-int(data[i*4+3]), H, W)
                    line = Line([y1, x1, y2, x2])
                    line.rescale(scale_H, scale_W)
                    lines.append(line)
            
            annotation = LineAnnotation(size=[args.fixsize, args.fixsize], lines=lines)

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
        # cv2.imwrite(save_name + '_p_label.jpg', hough_space_label*255)
        # cv2.imwrite(save_name + '_vis.jpg', vis)
        cv2.imwrite(save_name + '_mask.jpg', mask*255)
#print(stastic)
#print(angle_stastic.astype(np.int))

