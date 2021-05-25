import torch
import numpy as np
import math
import cv2 
import os
import torchvision
from PIL import Image
from basic_ops import *

def draw_line(y, x, angle, image, color=(0,0,255), num_directions=24):
    '''
    Draw a line with point y, x, angle in image with color.
    '''
    cv2.circle(image, (x, y), 2, color, 2)
    H, W = image.shape[:2]
    angle = int2arc(angle, num_directions)
    point1, point2 = get_boundary_point(y, x, angle, H, W)
    cv2.line(image, point1, point2, color, 2)
    return image

def convert_line_to_hough(line, size=(32, 32)):
    H, W = size
    theta = line.angle()
    alpha = theta + np.pi / 2
    if theta == -np.pi / 2:
        r = line.coord[1] - W/2
    else:
        k = np.tan(theta)
        y1 = line.coord[0] - H/2
        x1 = line.coord[1] - W/2
        r = (y1 - k*x1) / np.sqrt(1 + k**2)
    return alpha, r

def line2hough(line, numAngle, numRho, size=(32, 32)):
    H, W = size
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle

    r = int(np.round(r / irho)) + int((numRho) / 2)
    alpha = int(np.round(alpha / itheta))
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

def line2hough_float(line, numAngle, numRho, size=(32, 32)):
    H, W = size
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle

    r = r / irho + numRho / 2
    alpha = alpha / itheta
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    #return type: [(y1, x1, y2, x2)]
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []

    for (thetai, ri) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            # print('k = %.4f', - cosi / sini)
            # print('b = %.2f', np.round(r / sini + W * cosi / sini / 2 + H / 2))
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points

def visulize_mapping(b_points, size, filename):
    img = cv2.imread(os.path.join('./data/NKL', filename)) #change the path when using other dataset.
    img = cv2.resize(img, size)
    for (y1, x1, y2, x2) in b_points:
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=int(0.01*max(size[0], size[1])))
    return img

def caculate_precision(b_points, gt_coords, thresh=0.90):
    N = len(b_points)
    if N == 0:
        return 0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_p in enumerate(b_points):
        if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
            continue
        l_pred = Line(list(coord_p))
        for coord_g in gt_coords:
            l_gt = Line(list(coord_g))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N

def caculate_recall(b_points, gt_coords, thresh=0.90):
    N = len(gt_coords)
    if N == 0:
        return 1.0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_g in enumerate(gt_coords):
        l_gt = Line(list(coord_g))
        for coord_p in b_points:
            if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
                continue
            l_pred = Line(list(coord_p))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N

def coords_sort(coords):
    y1, x1, y2, x2 = coords
    if x1 > x2 or (x1 == x2 and y1 > y2):
        yy1, xx1, yy2, xx2 = y2, x2, y1, x1
    else:
        yy1, xx1, yy2, xx2 = y1, x1, y2, x2
    return yy1, xx1, yy2, xx2

def get_density(filename, x1, y1, x2, y2):
    hed_path = '/home/hanqi/JTLEE_code/pytorch-hed/hed_results/'
    filename = filename.split('_')[0]
    hed_file_path = os.path.join(hed_path, filename + '.png')
    hed = np.array(Image.open(hed_file_path).convert('L')) / 255

    mask = np.zeros_like(hed)
    mask = cv2.line(mask, (x1, y1), (x2, y2), color=1.0, thickness=7)

    density = (mask * hed).sum() / mask.sum()
    return density

def local_search(coords, coords_ring, d=1):

    y1, x1 = coords
    
    length = len(coords_ring)
    idx = coords_ring.index((x1, y1))
    new_x1, new_y1 = coords_ring[(idx + d) % length]

    return new_y1, new_x1 

def overflow(x, size=400):
    return x < 0 or x >= size

def edge_align(coords, filename, size, division=9):
    y1, x1, y2, x2 = coords
    ry1, rx1, ry2, rx2 = y1, x1, y2, x2
    if overflow(y1, size[0]) or overflow(x1, size[1]) or overflow(y2, size[0]) or overflow(x2, size[1]):
        return [ry1, rx1, ry2, rx2]
    density = 0
    hed_path = './data/sl6500_hed_results/'
    # hed_path = '/home/hanqi/JTLEE_code/pytorch-hed/hed_results/'
    filename = filename.split('.')[0]
    hed_file_path = os.path.join(hed_path, filename + '.png')
    hed = np.array(Image.open(hed_file_path).convert('L')) / 255
    
    coords_ring = [] #(x, y)
    #size = (400, 400)
    for i in range(0, size[1]):
        coords_ring.append((i, 0))
    for i in range(1, size[0]):
        coords_ring.append((size[1]-1, i))
    for i in range(size[1]-2, 0, -1):
        coords_ring.append((i, size[0]-1))
    for i in range(size[0]-1, 0, -1):
        coords_ring.append((0, i))


    for d1 in range(-division, division+1):
        for d2 in range(-division, division+1):
            ny1, nx1 = local_search([y1, x1], coords_ring, d=d1)
            ny2, nx2 = local_search([y2, x2], coords_ring, d=d2)

            mask = np.zeros_like(hed)
            mask = cv2.line(mask, (nx1, ny1), (nx2, ny2), color=1.0, thickness=3)
            dens = (mask * hed).sum() / mask.sum()
            if dens > density:
                density = dens
                ry1, rx1, ry2, rx2 = ny1, nx1, ny2, nx2

    return [ry1, rx1, ry2, rx2]
