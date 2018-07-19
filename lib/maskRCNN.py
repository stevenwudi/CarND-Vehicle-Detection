"""
maskRCNN: this is an mask-rcnn network for instance segmentation.
You need to download the repo from https://github.com/stevenwudi/CVPR_2018_WAD
TODO: verify the validity of the network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from collections import defaultdict
# Use a interactive backend
import sys
import matplotlib
matplotlib.use("TkAgg")

import cv2
import torch
import os.path as osp
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = '/home/stevenwudi/PycharmProjects/CVPR_2018_WAD'
# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
util_lib_path = osp.join(this_dir, 'utils')
add_path(util_lib_path)

import nn as mynn
from core.config import cfg, cfg_from_file, assert_and_infer_cfg

from modeling.model_builder import Generalized_RCNN
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018
import utils.net as net_utils
import utils.vis as vis_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

from lib.test_rle import im_detect_all, vis_one_image, binary_mask


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/e2e_mask_rcnn_R-101-FPN_2x.yaml', help='Config file for training (and optionally testing)')
    parser.add_argument('--load_ckpt', default='./Outputs/e2e_mask_rcnn_R-101-FPN_2x/Jun11-16-19-08_n606_step/ckpt/model_step8599.pth', help='checkpoint path to load')
    parser.add_argument('--dataset_dir', default='/media/samsumg_1tb/CVPR2018_WAD', help='directory to load images for demo')
    parser.add_argument('--cls_boxes_confident_threshold', type=float, default=0.1, help='threshold for detection boundingbox')
    parser.add_argument('--nms_soft', default=False, help='Using Soft NMS')
    parser.add_argument('--nms', default=0.5, help='default value for NMS')
    parser.add_argument('--vis', default=True)
    parser.add_argument('--range', default=None, help='start (inclusive) and end (exclusive) indices', type=int, nargs=2)
    args = parser.parse_args()
    return args


class maskRCNN:
    def __init__(self):
        args = parse_args()
        # Historical code, because we have trained on the Baidu Apoloscape dataset.
        dataset = WAD_CVPR2018(args.dataset_dir)
        cfg.MODEL.NUM_CLASSES = len(dataset.eval_class) + 1  # with a background class
        print('load cfg from file: {}'.format(args.cfg_file))
        cfg_from_file(osp.join(this_dir, args.cfg_file))
        if args.nms_soft:
            cfg.TEST.SOFT_NMS.ENABLED = True
        else:
            cfg.TEST.NMS = args.nms

        cfg.RESNETS.IMAGENET_PRETRAINED = False  # Don't need to load imagenet pretrained weights
        assert_and_infer_cfg()

        maskRCNN = Generalized_RCNN()
        maskRCNN.cuda()

        if args.load_ckpt:
            load_name = osp.join(this_dir, args.load_ckpt)
            print("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
            net_utils.load_ckpt(maskRCNN, checkpoint['model'])

        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])
        maskRCNN.eval()
        self.model = maskRCNN
        self.dataset = dataset

    def vehicleDetection(self, im):
        im_bgr = im[:, :, ::-1]  # RGB --> BGR for visualization
        cls_boxes, cls_segms, prediction_row = im_detect_all(self.model, im_bgr)

        return cls_boxes, cls_segms, prediction_row

    def vis(self, im, cls_boxes, cls_segms):

        vis_one_image(
            im,  # BGR -> RGB for visualization
            boxes=cls_boxes,
            segms=cls_segms,
            thresh=0.99,
            dataset=self.dataset,
            box_alpha=0.3,
            show_class=False,
        )

    def binary_mask(self, cls_boxes, cls_segms):
        return binary_mask(boxes=cls_boxes,
            segms=cls_segms,
            thresh=0.99,
        )



