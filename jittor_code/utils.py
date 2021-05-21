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
    # img = cv2.imread(os.path.join('./data/training/JTLEE_resize_100_100', filename))  #hanqi
    #img = cv2.resize(img, size)
    #scale_w = img.shape[1] / 400
    #scale_h = img.shape[0] / 400
    for (y1, x1, y2, x2) in b_points:
        #x1 = int(x1 * scale_w)
        #x2 = int(x2 * scale_w)
        #y1 = int(y1 * scale_h)
        #y2 = int(y2 * scale_h)
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

# def LinePooling(feat, coords, align_size=64):
#     '''
#     feat with shape [N, C, H, W]
#     coords with shape [N, K, 4], 4 means [y1, x1, y2, x2]
#     '''
#     N, C, H, W = feat.size()
#     scale = 1 / 4
#     K = coords.size(1)
#     assert K > 0
#     assert N == 1 # only support batch_size==1 in current version
#     for bs in range(N):
#         coord_st = coords[bs, :, 0:2] * scale
#         coord_ed = coords[bs, :, 2:4] * scale
#         with torch.no_grad():
#             arr_st2ed = coord_ed - coord_st # [K, 2]
#             sample_grid = torch.linspace(0, 1, align_size).to(feat).view(1, align_size).expand(K, align_size)
#             sample_grid = torch.einsum("id,is->isd", (arr_st2ed, sample_grid)) + coord_st.view(K, 1, 2).expand(K, align_size, 2)
#             sample_grid = sample_grid.view(K, align_size, 1, 2)
#             sample_grid[..., 0] = sample_grid[..., 0] / (H - 1) * 2 - 1
#             sample_grid[..., 1] = sample_grid[..., 1] / (W - 1) * 2 - 1
#         output = torch.nn.functional.grid_sample(feat[int(bs)].view(1, C, H, W).expand(K, C, H, W), sample_grid)
#         output = output.view(K, C, align_size)
    
#     return output

# def gen_proposal(gt_coord, size=(400, 400)):
#     y1, x1, y2, x2 = gt_coord
#     if y1 == 0 or y1 == size[0] - 1:
#         delta_y1 = 0
#         delta_x1 = np.random.uniform(-0.02, 0.02) * size[1]
#         if x1 + delta_x1 >= size[1]:
#             delta_x1 = size[1] - 1 - x1
#         elif x1 + delta_x1 <= 0:
#             delta_x1 = 0 - x1
#     else:
#         delta_y1 = np.random.uniform(-0.02, 0.02) * size[0]
#         delta_x1 = 0
#         if y1 + delta_y1 >= size[0]:
#             delta_y1 = size[0] - 1 - y1
#         elif y1 + delta_y1 <= 0:
#             delta_y1 = 0 - y1
    
#     if y2 == 0 or y2 == size[0] - 1:
#         delta_y2 = 0
#         delta_x2 = np.random.uniform(-0.02, 0.02) * size[1]
#         if x2 + delta_x2 >= size[1]:
#             delta_x2 = size[1] - 1 - x2
#         elif x2 + delta_x2 <= 0:
#             delta_x2 = 0 - x2
#     else:
#         delta_y2 = np.random.uniform(-0.02, 0.02) * size[0]
#         delta_x2 = 0
#         if y2 + delta_y2 >= size[0]:
#             delta_y2 = size[0] - 1 - y2
#         elif y2 + delta_y2 <= 0:
#             delta_y2 = 0 - y2
#     return delta_y1, delta_x1, delta_y2, delta_x2

#coords_ring = [] #(x, y)
#size = (400, 400)
#for i in range(0, size[1]):
#    coords_ring.append((i, 0))
#for i in range(1, size[0]):
#    coords_ring.append((size[1]-1, i))
#for i in range(size[1]-2, 0, -1):
#    coords_ring.append((i, size[0]-1))
#for i in range(size[0]-1, 0, -1):
#    coords_ring.append((0, i))


# def gen_proposal_consist(gt_coord):
#     global coords_ring
#     y1, x1, y2, x2 = gt_coord
    
#     step1 = int(np.random.uniform(-5, 6))
#     step2 = int(np.random.uniform(-5, 6))
    
#     length = len(coords_ring)
#     idx = coords_ring.index((x1, y1))
#     new_x1, new_y1 = coords_ring[(idx + step1) % length]
#     idx = coords_ring.index((x2, y2))
#     new_x2, new_y2 = coords_ring[(idx + step2) % length]

#     return new_y1-y1, new_x1-x1, new_y2-y2, new_x2-x2, step1, step2

# def generate_proposals(gt_coords, size=(400, 400)):
#     # n_samples = 12 // len(gt_coords)
#     # n_samples = max(n_samples, 4)
#     n_samples = 4
#     proposals = []
#     labels = []
#     scale = 120
#     for i, gt_coord in enumerate(gt_coords):
#         # if i > 3: break
#         y1, x1, y2, x2 = gt_coord
#         for _ in range(n_samples):
#         # for j in range(1, 5):
#             # delta_y1, delta_x1, delta_y2, delta_x2 = gen_proposal(gt_coord, size)
#             delta_y1, delta_x1, delta_y2, delta_x2, step1, step2 = gen_proposal_consist(gt_coord)
            
#             proposals.append([y1+delta_y1, x1+delta_x1, y2+delta_y2, x2+delta_x2])
#             '''
#             dy1 = -delta_y1 / size[0] * scale            
#             dx1 = -delta_x1 / size[1] * scale
#             dy2 = -delta_y2 / size[0] * scale
#             dx2 = -delta_x2 / size[1] * scale
#             if abs(delta_y1) < 0.001:
#                 dy1 = 0.5
#             if abs(delta_x1) < 0.001:
#                 dx1 = 0.5
#             if abs(delta_y2) < 0.001:
#                 dy2 = 0.5
#             if abs(delta_x2) < 0.001:
#                 dx2 = 0.5                
#             '''
#             labels.append([-delta_y1 / size[0] * scale, -delta_x1 / size[1] * scale, -delta_y2 / size[0] * scale, -delta_x2 / size[1] * scale])
#             #labels.append([step1 / 15, step2 / 15])
#             #labels.append([dy1, dx1, dy2, dx2])
#             #labels.append([-delta_y1 / size[0] * scale, -delta_x1 / size[1] * scale, -delta_y2 / size[0] * scale, -delta_x2 / size[1] * scale])
#             #labels.append([-delta_y1 * scale, -delta_x1 * scale, -delta_y2  * scale, -delta_x2  * scale])
#         delta_y1, delta_x1, delta_y2, delta_x2 = 0, 0, 0, 0
#         proposals.append([y1+delta_y1, x1+delta_x1, y2+delta_y2, x2+delta_x2])
#         labels.append([-delta_y1 / size[0] * scale, -delta_x1 / size[1] * scale, -delta_y2 / size[0] * scale, -delta_x2 / size[1] * scale])
#     return proposals, labels
        
# def add_delta(coords, deltas, size=(400, 400)):
#     scale = 120
#     deltas = deltas * 400 / scale

#     y1, x1, y2, x2 = coords
    
#     if left(x1, y1, size) or right(x1, y1, size):
#         y1 += deltas[0]
#     else:
#         x1 += deltas[1]
#     if left(x2, y2, size) or right(x2, y2, size):
#         y2 += deltas[2]
#     else:
#         x2 += deltas[3]
   
#     return y1, x1, y2, x2

# def make_labels(b_points, gt_coords):
#     size = (400, 400)
#     scale = 40
#     pred_lines = [Line(list(coord)) for coord in b_points]
#     gt_lines = []
#     for i, coord_g in enumerate(gt_coords):
#         if coord_g[0]==coord_g[2] and coord_g[1]==coord_g[3]:
#                 continue
#         gt_lines.append(Line(list(coord_g)))
        
    
#     labels = []
#     for i in range(len(pred_lines)):
#         metrics = [EA_metric(pred_lines[i], gt_line) for gt_line in gt_lines]
#         idx = np.argmax(metrics)
#         delta_y1 = b_points[i][0] - gt_coords[idx][0]
#         delta_x1 = b_points[i][1] - gt_coords[idx][1]
#         delta_y2 = b_points[i][2] - gt_coords[idx][2]
#         delta_x2 = b_points[i][3] - gt_coords[idx][3]
    
#         labels.append([-delta_y1 / size[0] * scale, -delta_x1 / size[1] * scale, -delta_y2 / size[0] * scale, -delta_x2 / size[1] * scale])
#     return labels

def coords_sort(coords):
    y1, x1, y2, x2 = coords
    if x1 > x2 or (x1 == x2 and y1 > y2):
        yy1, xx1, yy2, xx2 = y2, x2, y1, x1
    else:
        yy1, xx1, yy2, xx2 = y1, x1, y2, x2
    return yy1, xx1, yy2, xx2

# def left(x1, y1, size):
#     if x1 == 0:
#         return True
#     else:  
#         return False

# def right(x1, y1, size):
#     if x1 == size[1] - 1:
#         return True
#     else:
#         return False

# def top(x1, y1, size):
#     if y1 == 0:
#         return True
#     else:
#         return False

# def bottom(x1, y1, size):
#     if y1 == size[0] - 1:
#         return True
#     else:
#         return False

# def generate_proposals_parametric(gt_coords, size=(400, 400)):
#     output_lr = -np.ones((200, 200), dtype=np.float32)
#     output_ud = -np.ones((200, 200), dtype=np.float32)
#     # mask = np.zeros((100, 100), dtype=np.float32)
#     for i, gt_coord in enumerate(gt_coords):
#         y1, x1, y2, x2 = gt_coord
#         if y1 == y2 and x1 == x2:
#             continue
#         gt_l = Line([y1, x1, y2, x2])
#         theta, r = line2hough(gt_l, 200, 200, size)

#         for i in range(theta-50, theta+51):
#             for j in range(r-50, r+51):
#                 if i < 0 or i >= 200:
#                     continue
#                 if j < 0 or j >= 200:
#                     continue
#                 # if i == theta and j == r:
#                 #     continue
#                 # if i != theta:
#                 #     mask[0, i, j] = 1
#                 # if j != r:
#                 #     mask[1, i, j] = 1
#                 if i < theta:
#                     output_lr[i, j] = 0
#                 elif i > theta:
#                     output_lr[i, j] = 1
#                 else:
#                     output_lr[i, j] = -1
                
#                 if j < r:
#                     output_ud[i, j] = 0
#                 elif i > theta:
#                     output_ud[i, j] = 1
#                 else:
#                     output_ud[i, j] = -1
#                 # output[0, i, j] = (i<theta) * 1.0
                # output[1, i, j] = (j<r) * 1.0

        # if theta-1>=0:
        #     mask[theta-1, r] = 1
        #     output[0, theta-1, r] = 0
        # if theta+1 < 100:
        #     mask[theta+1, r] = 1
        #     output[0, theta+1, r] = 1
        # if r-1>=0:
        #     mask[theta, r-1] = 1
        #     output[1, theta, r-1] = 0
        # if r+1 < 100:
        #     mask[theta, r+1] = 1
        #     output[1, theta, r+1] = 1
        
        # if theta-2>=0:
        #     mask[theta-2, r] = 1
        #     output[0, theta-2, r] = 0
        # if theta+2<100:
        #     mask[theta+2, r] = 1
        #     output[0, theta+2, r] = 1
        # if r-2>=0:
        #     mask[theta, r-2] = 1
        #     output[1, theta, r-2] = 0
        # if r+2<100:
        #     mask[theta, r+2] = 1
        #     output[1, theta, r+2] = 1

        # if theta-3>=0:
        #     mask[theta-3, r] = 1
        #     output[0, theta-3, r] = 0
        # if theta+3<100:
        #     mask[theta+3, r] = 1
        #     output[0, theta+3, r] = 1
        # if r-3>=0:
        #     mask[theta, r-3] = 1
        #     output[1, theta, r-3] = 0
        # if r+3<100:
        #     mask[theta, r+3] = 1
        #     output[1, theta, r+3] = 1
    
    # return output_lr, output_ud


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
