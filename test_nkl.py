import argparse 
import numpy as np
import os
from hungarian_matching import caculate_tp_fp_fn

gt_path = './data/training/SL5K_resize_100_100'
#gt_path = '/home/hanqi/work/semantic/data/crawl/JTLEE_resize_100_100'
parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
parser.add_argument('--pred', type=str, required=True)
parser.add_argument('--gt', type=str, required=True)
parser.add_argument('--align', default=False, action='store_true')
arg = parser.parse_args()
#
pred_path = arg.pred 
gt_path = arg.gt
filenames = sorted(os.listdir(pred_path))

total_tp = np.zeros(99)
total_fp = np.zeros(99)
total_fn = np.zeros(99)

total_tp_align = np.zeros(99)
total_fp_align = np.zeros(99)
total_fn_align = np.zeros(99)

for filename in filenames:
    if 'npy' not in filename:
        continue
    if 'align' in filename:
        continue
    pred = np.load(os.path.join(pred_path, filename))
    if arg.align:
        pred_align = np.load(os.path.join(pred_path, filename.split('.')[0]+'_align.npy'))
    gt_txt = open(os.path.join(gt_path, filename.split('.')[0] + '.txt'))
    gt_coords = gt_txt.readlines()
    gt = []
    for i in range(int(gt_coords[0].split(' ')[0])):
        gt.append([int(float(gt_coords[0].split(' ')[i*4+2])), int(float(gt_coords[0].split(' ')[i*4+1])), int(float(gt_coords[0].split(' ')[i*4+4])), int(float(gt_coords[0].split(' ')[i*4+3]))])
    for i in range(1, 100):
        tp, fp, fn = caculate_tp_fp_fn(pred.tolist(), gt, thresh=i*0.01)
        total_tp[i-1] += tp
        total_fp[i-1] += fp
        total_fn[i-1] += fn
        if arg.align:
            tp, fp, fn = caculate_tp_fp_fn(pred_align.tolist(), gt, thresh=i*0.01)
            total_tp_align[i-1] += tp
            total_fp_align[i-1] += fp
            total_fn_align[i-1] += fn
        

    
total_recall = total_tp / (total_tp + total_fn)
total_precision = total_tp / (total_tp + total_fp)
f = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-6)

if arg.align:
    total_recall_align = total_tp_align / (total_tp_align + total_fn_align)
    total_precision_align = total_tp_align / (total_tp_align + total_fp_align)
    f_align = 2 * total_recall_align * total_precision_align / (total_recall_align + total_precision_align + 1e-6)

print('Mean P:', total_precision.mean())
print('Mean R:', total_recall.mean())
print('Mean F:', f.mean()) 
print('F@0.95:', f[94])
#np.savetxt('precision.csv', total_precision)
#np.savetxt('recall.csv', total_recall)
#np.savetxt('fscore.csv', total_f)

if arg.align:
    print('Mean P_align:', total_precision_align.mean())
    print('Mean R_align:', total_recall_align.mean())
    print('Mean F_align:', f_align.mean()) 
    print('F_align@0.95:', f_align[94])
    #np.savetxt('total_precision_refine.csv', total_precision_align)
    #np.savetxt('total_recall_refine.csv', total_recall_align)
    #np.savetxt('total_f_refine.csv', f_align)
