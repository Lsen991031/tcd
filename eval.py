import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import datetime
import time
import os
import csv

from ops.models import TSN
from ops.dataset import TSNDataSet, TSNDataSetBoth
from ops.transforms import *
from ops.utils import * #AverageMeter, accuracy
from cl_methods.classifer import nme, compute_class_mean
import copy
from opts import parser
import shutil

def eval_task(args,):
    # Construct TSM Models
    model = TSN(args, num_class=2, modality='RGB',
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                age=25,cur_task_size=2)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    ckpt_path = os.path.join('checkpoint', 'ucf101', str(51),  str(2), '{:03d}'.format(0), 'task_{:03d}.pth.tar'.format(25))
    print("Load the Trained Model from {}".format(ckpt_path))
    sd = torch.load(ckpt_path)
    sd = sd['state_dict']
    model.load_state_dict(sd)
    model.eval()

    # get input
    #input_vedio = 


    # compute output
    #outputs = model(input_video)
    #output = outputs['preds']
    #del outputs

    #return output

if __name__ == '__main__':
    args = parser.parse_args()
    eval_task(args)