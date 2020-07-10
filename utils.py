import numpy as np
import cv2 
import os
from metric import EA_metric
from basic_ops import Line, get_boundary_point

def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
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
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points

def visulize_mapping(b_points, size, filename):
    img = cv2.imread(filename) 
    #img = cv2.resize(img, size)
    for (y1, x1, y2, x2) in b_points:
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=3)
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
