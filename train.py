import argparse
import os
import random
import shutil
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim
import tqdm
import yaml
from torch.optim import lr_scheduler
from logger import Logger

from dataloader import get_loader
from model.network import Net
from skimage.measure import label, regionprops
from tensorboardX import SummaryWriter
from utils import reverse_mapping, edge_align
from hungarian_matching import caculate_tp_fp_fn

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
# arguments from command line
parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--resume', default="", help='path to config file')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.load(open(args.config))

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"] = float(CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"])
CONFIGS["OPTIMIZER"]["LR"] = float(CONFIGS["OPTIMIZER"]["LR"])

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))



logger.info(CONFIGS)

def main():

    logger.info(args)
    assert os.path.isdir(CONFIGS["DATA"]["DIR"])

    if CONFIGS['TRAIN']['SEED'] is not None:
        random.seed(CONFIGS['TRAIN']['SEED'])
        torch.manual_seed(CONFIGS['TRAIN']['SEED'])
        cudnn.deterministic = True

    model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])
    
    if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
        logger.info("Model Data Parallel")
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIGS["OPTIMIZER"]["LR"],
        weight_decay=CONFIGS["OPTIMIZER"]["WEIGHT_DECAY"]
    )

    # learning rate scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                            milestones=CONFIGS["OPTIMIZER"]["STEPS"],
                            gamma=CONFIGS["OPTIMIZER"]["GAMMA"])
    best_acc1 = 0
    if args.resume:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # dataloader
    train_loader = get_loader(CONFIGS["DATA"]["DIR"], CONFIGS["DATA"]["LABEL_FILE"], 
                                batch_size=CONFIGS["DATA"]["BATCH_SIZE"], num_thread=CONFIGS["DATA"]["WORKERS"], split='train')
    val_loader = get_loader(CONFIGS["DATA"]["VAL_DIR"], CONFIGS["DATA"]["VAL_LABEL_FILE"], 
                                batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], split='val')

    logger.info("Data loading done.")

    # Tensorboard summary

    writer = SummaryWriter(log_dir=os.path.join(CONFIGS["MISC"]["TMP"]))

    start_epoch = 0
    best_acc = best_acc1
    is_best = False
    start_time = time.time()

    if CONFIGS["TRAIN"]["RESUME"] is not None:
        raise(NotImplementedError)
    
    if CONFIGS["TRAIN"]["TEST"]:
        validate(val_loader, model, 0, writer, args)
        return

    logger.info("Start training.")

    for epoch in range(start_epoch, CONFIGS["TRAIN"]["EPOCHS"]):
        
        train(train_loader, model, optimizer, epoch, writer, args)
        acc = validate(val_loader, model, epoch, writer, args)
        #return
        scheduler.step()

        if best_acc < acc:
            is_best = True
            best_acc = acc
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=CONFIGS["MISC"]["TMP"])

        t = time.time() - start_time       
        elapsed = DayHourMinute(t)
        t /= (epoch + 1) - start_epoch    # seconds per epoch
        t = (CONFIGS["TRAIN"]["EPOCHS"] - epoch - 1) * t
        remaining = DayHourMinute(t)
        
        logger.info("Epoch {0}/{1} finishied, auxiliaries saved to {2} .\t"
                    "Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes.\t"
                    "Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.".format(
                    epoch, CONFIGS["TRAIN"]["EPOCHS"], CONFIGS["MISC"]["TMP"], elapsed=elapsed, remaining=remaining))

    logger.info("Optimization done, ALL results saved to %s." % CONFIGS["MISC"]["TMP"])

def train(train_loader, model, optimizer, epoch, writer, args):
    # switch to train mode
    model.train()
    # torch.cuda.empty_cache()
    bar = tqdm.tqdm(train_loader)
    iter_num = len(train_loader.dataset) // CONFIGS["DATA"]["BATCH_SIZE"]

    total_loss_hough = 0
    for i, data in enumerate(bar):

        images, hough_space_label, _, names = data

        if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
            images = images.cuda()
            hough_space_label = hough_space_label.cuda()
        else:
            images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            hough_space_label = hough_space_label.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            
        keypoint_map = model(images)

        hough_space_loss = torch.nn.functional.binary_cross_entropy_with_logits(keypoint_map, hough_space_label)

        writer.add_scalar('train/hough_space_loss', hough_space_loss.item(), epoch * iter_num + i)

        loss = hough_space_loss

        if not torch.isnan(hough_space_loss):
            total_loss_hough += hough_space_loss.item()
        else:
            logger.info("Warnning: loss is Nan.")

        #record loss
        bar.set_description('Training Loss:{}'.format(loss.item()))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % CONFIGS["TRAIN"]["PRINT_FREQ"] == 0:
            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize', str(epoch))
            os.makedirs(visualize_save_path, exist_ok=True)
            
            # Do visualization.
            # torchvision.utils.save_image(torch.sigmoid(keypoint_map), join(visualize_save_path, 'rodon_'+names[0]), normalize=True)
            # torchvision.utils.save_image(torch.sum(vis, dim=1, keepdim=True), join(visualize_save_path, 'vis_'+names[0]), normalize=True)

    total_loss_hough = total_loss_hough / iter_num
    writer.add_scalar('train/total_loss_hough', total_loss_hough, epoch)
 

    
def validate(val_loader, model, epoch, writer, args):
    # switch to evaluate mode
    model.eval()
    total_acc = 0.0
    total_loss_hough = 0

    total_tp = np.zeros(99)
    total_fp = np.zeros(99)
    total_fn = np.zeros(99)

    total_tp_align = np.zeros(99)
    total_fp_align = np.zeros(99)
    total_fn_align = np.zeros(99)

    with torch.no_grad():
        bar = tqdm.tqdm(val_loader)
        iter_num = len(val_loader.dataset) // 1
        for i, data in enumerate(bar):

            images, hough_space_label8, gt_coords, names = data

            if CONFIGS["TRAIN"]["DATA_PARALLEL"]:
                images = images.cuda()
                hough_space_label8 = hough_space_label8.cuda()
            else:
                images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
                hough_space_label8 = hough_space_label8.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
                
            keypoint_map = model(images)

            hough_space_loss = torch.nn.functional.binary_cross_entropy_with_logits(keypoint_map, hough_space_label8)
            writer.add_scalar('val/hough_space_loss', hough_space_loss.item(), epoch * iter_num + i)

            acc = 0
            total_acc += acc

            loss = hough_space_loss
            if not torch.isnan(loss):
                total_loss_hough += loss.item()
            else:
                logger.info("Warnning: val loss is Nan.")

            key_points = torch.sigmoid(keypoint_map)
            binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)
            props = regionprops(kmap_label)
            plist = []
            for prop in props:
                plist.append(prop.centroid)
            b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))
            # [[y1, x1, y2, x2], [] ...]
            gt_coords = gt_coords[0].tolist()
            for i in range(1, 100):
                tp, fp, fn = caculate_tp_fp_fn(b_points, gt_coords, thresh=i*0.01)
                total_tp[i-1] += tp
                total_fp[i-1] += fp
                total_fn[i-1] += fn

            if CONFIGS["MODEL"]["EDGE_ALIGN"]:
                for i in range(len(b_points)):
                    b_points[i] = edge_align(b_points[i], names[0], division=5)
                
                for i in range(1, 100):
                    tp, fp, fn = caculate_tp_fp_fn(b_points, gt_coords, thresh=i*0.01)
                    total_tp_align[i-1] += tp
                    total_fp_align[i-1] += fp
                    total_fn_align[i-1] += fn
            
        total_loss_hough = total_loss_hough / iter_num
        
        total_recall = total_tp / (total_tp + total_fn + 1e-8)
        total_precision = total_tp / (total_tp + total_fp + 1e-8)
        f = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-8)
        
       
        writer.add_scalar('val/total_loss_hough', total_loss_hough, epoch)
        writer.add_scalar('val/total_precison', total_precision.mean(), epoch)
        writer.add_scalar('val/total_recall', total_recall.mean(), epoch)
        logger.info('Validation result: ==== Precision: %.5f, Recall: %.5f' % (total_precision.mean(), total_recall.mean()))
        acc = f.mean()
        logger.info('Validation result: ==== F-measure: %.5f' % acc.mean())
        logger.info('Validation result: ==== F-measure@0.95: %.5f' % f[95 - 1])
        writer.add_scalar('val/f-measure', acc.mean(), epoch)
        writer.add_scalar('val/f-measure@0.95', f[95 - 1], epoch)
        
        if CONFIGS["MODEL"]["EDGE_ALIGN"]:
            total_recall_align = total_tp_align / (total_tp_align + total_fn_align + 1e-8)
            total_precision_align = total_tp_align / (total_tp_align + total_fp_align + 1e-8)
            f_align = 2 * total_recall_align * total_precision_align / (total_recall_align + total_precision_align + 1e-8)
            writer.add_scalar('val/total_precison_align', total_precision_align.mean(), epoch)
            writer.add_scalar('val/total_recall_align', total_recall_align.mean(), epoch)
            logger.info('Validation result (Aligned): ==== Precision: %.5f, Recall: %.5f' % (total_precision_align.mean(), total_recall_align.mean()))
            acc = f_align.mean()
            logger.info('Validation result (Aligned): ==== F-measure: %.5f' % acc.mean())
            logger.info('Validation result (Aligned): ==== F-measure@0.95: %.5f' % f_align[95 - 1])
            writer.add_scalar('val/f-measure', acc.mean(), epoch)
            writer.add_scalar('val/f-measure@0.95', f_align[95 - 1], epoch)
    return acc.mean()


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth'))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



class DayHourMinute(object):
  
  def __init__(self, seconds):
      
      self.days = int(seconds // 86400)
      self.hours = int((seconds- (self.days * 86400)) // 3600)
      self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)


if __name__ == '__main__':
    main()
