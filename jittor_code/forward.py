import argparse
import os
import time
from os.path import isfile, join, split

import numpy as np
import tqdm
import yaml
import cv2
import jittor as jt
from jittor import nn

from logger import Logger

from dataloader import get_loader
from model.network import Net
from skimage.measure import label, regionprops
from utils import reverse_mapping, visulize_mapping, get_boundary_point

if jt.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description='Jittor Semantic-Line Inference')
parser.add_argument('--config', default="../config.yml", help="path to config file")
parser.add_argument('--model', required=True, help='path to the pretrained model')
parser.add_argument('--tmp', default="", help='tmp')
parser.add_argument('--dump', default=False)
args = parser.parse_args()

assert os.path.isfile(args.config)
CONFIGS = yaml.load(open(args.config))

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))

def main():

    logger.info(args)

    model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])

    if args.model:
        if isfile(args.model):
            logger.info("=> loading pretrained model '{}'".format(args.model))
            import torch
            m = torch.load(args.model)
            if 'state_dict' in m.keys():
                m = m['state_dict']
            torch.save(m, '_temp_model.pth')
            del m
            model.load('_temp_model.pth')
            logger.info("=> loaded checkpoint '{}'".format(args.model))
        else:
            logger.info("=> no pretrained model found at '{}'".format(args.model))
    # dataloader
    test_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"], 
                                batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True)
    logger.info("Data loading done.")
    
    weights_nodes = {}
    data_nodes = {}

    def named_dump_func(name):
        def dump_func(self, inputs, outputs):
            input_name = name + '_input'
            output_name = name + '_output'
            if isinstance(self, nn.Conv2d):
                weights_nodes[name] = self.weight.numpy()
            data_nodes[input_name] = inputs[0].numpy()
            data_nodes[output_name] = outputs[0].numpy()
        return dump_func

    if args.dump:
        logger.info('Add hooks to dump data.')
        for name, module in model.named_modules():
            module.register_forward_hook(named_dump_func(name))

    logger.info("Start testing.")
    total_time = test(test_loader, model, args)

    if args.dump:
        np.save('data_nodes.npy', data_nodes)
        np.save('weights_nodes.npy', weights_nodes)
        exit()

    logger.info("Test done! Total %d imgs at %.4f secs without image io, fps: %.3f" % (len(test_loader), total_time, len(test_loader) / total_time))

        
def test(test_loader, model, args):
    # switch to evaluate mode
    model.eval()
    bar = tqdm.tqdm(test_loader)
    ftime = 0
    ttime = 0
    ntime = 0
    for i, data in enumerate(bar):
        t = time.time()
        images, names, size = data
        
        images = jt.array(images)
        # size = (size[0].item(), size[1].item())       
        key_points = model(images)

        if args.dump:
            break

        key_points = key_points.sigmoid()
        ftime += (time.time() - t)

        visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
        os.makedirs(visualize_save_path, exist_ok=True)

        binary_kmap = key_points.squeeze(0).squeeze(0).numpy() > CONFIGS['MODEL']['THRESHOLD']
        kmap_label = label(binary_kmap, connectivity=1)
        props = regionprops(kmap_label)
        plist = []
        for prop in props:
            plist.append(prop.centroid)

        size = (size[0][0], size[0][1])
        b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))
        scale_w = size[1] / 400
        scale_h = size[0] / 400
        for i in range(len(b_points)):
            y1 = int(np.round(b_points[i][0] * scale_h))
            x1 = int(np.round(b_points[i][1] * scale_w))
            y2 = int(np.round(b_points[i][2] * scale_h))
            x2 = int(np.round(b_points[i][3] * scale_w))
            if x1 == x2:
                angle = -np.pi / 2
            else:
                angle = np.arctan((y1-y2) / (x1-x2))
            (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
            b_points[i] = (y1, x1, y2, x2)
        ttime += (time.time() - t)
        
        vis = visulize_mapping(b_points, size, names[0])

        

        cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1]), vis)
        np_data = np.array(b_points)
        np.save(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]), np_data)

        ntime += (time.time() - t)
    if args.dump:
        return 0
    print('forward fps for total images: %.6f' % (len(test_loader) / ftime))
    print('forward + post-processing fps for total images: %.6f' % (len(test_loader) / ttime))
    print('total fps for total images: %.6f' % (len(test_loader) / ntime))
    return ntime

if __name__ == '__main__':
    main()
