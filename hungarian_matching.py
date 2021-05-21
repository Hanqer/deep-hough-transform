import numpy as np 
from scipy.optimize import linear_sum_assignment
# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
from metric import EA_metric, Chamfer_metric, Emd_metric
from basic_ops import *

def build_graph(p_lines, g_lines, threshold):
    prediction_len = len(p_lines)
    gt_len = len(g_lines)
    G = np.zeros((prediction_len, gt_len))
    for i in range(prediction_len):
        for j in range(gt_len):
            if EA_metric(p_lines[i], g_lines[j]) >= threshold:
            # if Chamfer_metric(p_lines[i], g_lines[j]) >= threshold:
            # if Emd_metric(p_lines[i], g_lines[j]) >= threshold:
                G[i][j] = 1
    return G

def caculate_tp_fp_fn(b_points, gt_coords, thresh=0.90):
    p_lines = []
    g_lines = []
    for points in b_points:
        if len(points) == 0:
            continue 
        if points[0] == points[2] and points[1] == points[3]:
            continue 
        else:
            p_lines.append(Line(list(points)))
    
    for points in gt_coords:
        if len(points) == 0:
            continue
        if points[0] == points[2] and points[1] == points[3]:
            continue 
        else:
            g_lines.append(Line(list(points)))
    
    G = build_graph(p_lines, g_lines, thresh)
    # convert G to -G to caculate maximum matching.
    row_ind, col_ind = linear_sum_assignment(-G)

    pair_nums = G[row_ind, col_ind].sum()

    tp = pair_nums
    fp = len(p_lines) - pair_nums
    fn = len(g_lines) - pair_nums
    return tp, fp, fn
