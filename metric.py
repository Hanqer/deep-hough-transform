import numpy as np 
import cv2
import torch
import ot
from basic_ops import Line
from chamfer_distance import ChamferDistance

cd = ChamferDistance()

def sa_metric(angle_p, angle_g):
    d_angle = np.abs(angle_p - angle_g)
    d_angle = min(d_angle, np.pi - d_angle)
    d_angle = d_angle * 2 / np.pi
    return max(0, (1 - d_angle)) ** 2

def se_metric(coord_p, coord_g, size=(400, 400)):
    c_p = [(coord_p[0] + coord_p[2]) / 2, (coord_p[1] + coord_p[3]) / 2]
    c_g = [(coord_g[0] + coord_g[2]) / 2, (coord_g[1] + coord_g[3]) / 2]
    d_coord = np.abs(c_p[0] - c_g[0])**2 + np.abs(c_p[1] - c_g[1])**2
    d_coord = np.sqrt(d_coord) / max(size[0], size[1])
    return max(0, (1 - d_coord)) ** 2

def EA_metric(l_pred, l_gt, size=(400, 400)):
    se = se_metric(l_pred.coord, l_gt.coord, size=size)
    sa = sa_metric(l_pred.angle(), l_gt.angle())
    return sa * se

def Chamfer_metric(l_pred, l_gt, size=(400, 400)):
    points1 = get_points_coords(l_pred)
    points2 = get_points_coords(l_gt)
    #add z-axis
    points1 = np.insert(points1, 0, values=0, axis=1)
    points2 = np.insert(points2, 0, values=0, axis=1)

    
    p1 = torch.from_numpy(points1).unsqueeze(0).float()
    p2 = torch.from_numpy(points2).unsqueeze(0).float()

    d1, d2 = cd(p1, p2)
    
    d = (d1.mean().item() + d2.mean().item()) / 2
    mmax = size[0] * size[0] + size[1] * size[1]
    
    return 1 - d / mmax

def Emd_metric(l_pred, l_gt, size=(400, 400)):
    points1 = get_points_coords(l_pred)
    points2 = get_points_coords(l_gt)

    M = ot.dist(points1, points2, metric='euclidean')

    _, log = ot.emd([], [], M, log=True)
    cost = log['cost']
    return 1 - cost / np.sqrt(size[0] * size[0] + size[1] * size[1])

def get_points_coords(l):
    points = []
    y0, x0, y1, x1 = l.coord
    dx = x1 - x0
    dy = y1 - y0
    length = int(np.sqrt(dx * dx + dy * dy))
    for _ in range(length + 1):
        points.append([int(np.round(x0)), int(np.round(y0))])
        x0 += (dx / length)
        y0 += (dy / length)
    return points


    
if __name__ == "__main__":
    # l1 = Line([0, 200, 400, 200])
    # l2 = Line([200, 0, 200, 400])

    l1 = Line([200, 0, 190, 399])
    l2 = Line([190, 0, 200, 399])
    print(EA_metric(l1, l2))

    mask = np.zeros((400, 400))
    cv2.line(mask, (5, 0), (0, 5), 255, 1)
    cv2.line(mask, (394, 399), (399, 394), 255, 1)
    cv2.imwrite('debug.png', mask)
    cd_score = Chamfer_metric(l1, l2)
    emd_score = Emd_metric(l1, l2)
    print(cd_score, emd_score)
    
