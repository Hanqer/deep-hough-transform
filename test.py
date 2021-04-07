import argparse 
import numpy as np
import os
from utils import caculate_precision, caculate_recall

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
parser.add_argument('--pred', type=str, required=True)
parser.add_argument('--gt', type=str, required=True)
arg = parser.parse_args()
#
pred_path = arg.pred 
gt_path = arg.gt
filenames = sorted(os.listdir(pred_path))

total_precision = np.zeros(99)
total_recall = np.zeros(99)
nums_precision = 0
nums_recall = 0
for filename in filenames:
    if 'npy' not in filename:
        continue
    pred = np.load(os.path.join(pred_path, filename))
    gt_txt = open(os.path.join(gt_path, filename.split('.')[0] + '.txt'))
    gt_coords = gt_txt.readlines()
    gt = [[int(float(l.rstrip().split(', ')[1])), int(float(l.rstrip().split(', ')[0])), int(float(l.rstrip().split(', ')[3])), int(float(l.rstrip().split(', ')[2]))] for l in gt_coords]

    for i in range(1, 100):
        p, num_p = caculate_precision(pred.tolist(), gt, thresh=i*0.01)
        r, num_r = caculate_recall(pred.tolist(), gt, thresh=i*0.01)
        total_precision[i-1] += p
        total_recall[i-1] += r
    nums_precision += num_p
    nums_recall += num_r
    
total_recall = total_recall / 298
total_precision = total_precision / nums_precision
f = 2 * total_recall * total_precision / (total_recall + total_precision)

print('Mean P:', total_precision.mean())
print('Mean R:', total_recall.mean())
print('Mean F:', f.mean()) 