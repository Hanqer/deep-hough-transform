import argparse
import os
import random
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
parser.add_argument('--config', default="./config.yml", help="path to config file")
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
            import torch 
            m = torch.load(args.model)
            if 'state_dict' in m.keys():
                m = m['state_dict']
            torch.save(m, '_temp_model.pth')
            del m 
            logger.info("=> loading pretrained model '{}'".format(args.model))
            #model.load('_temp_model.pth')
            logger.info("=> loaded checkpoint '{}'".format(args.model))
        else:
            logger.info("=> no pretrained model found at '{}'".format(args.model))
    # dataloader
    test_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"], 
                                batch_size=int(os.environ.get("BS","1")), num_thread=CONFIGS["DATA"]["WORKERS"], test=True)
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
            print(name)
            module.register_forward_hook(named_dump_func(name))

    test(test_loader, model, args)

        
@jt.no_grad()
def test(test_loader, model, args):
    # switch to evaluate mode
    model.eval()
    for data in test_loader:
        images, names, size = data
        break
    jt.sync_all(True)

    # warmup
    for i in range(10):
        model(images).sync()
    jt.sync_all(True)

    # rerun
    t = time.time()
    for i in range(300):
        print(i, i/(time.time()-t))
        model(images).sync()
    jt.sync_all(True)
    t = time.time()-t
    print("BS:", images.shape[0], "FPS:", 300*images.shape[0]/t)

if __name__ == '__main__':
    main()
