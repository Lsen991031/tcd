#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import torch
import torch.nn as nn
import os
import cv2
import matplotlib.cm as cm
'''
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
'''

import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from models.build import build_model, increment_model
import models.optimizer as optim
from utils.meters import AVAMeter, TestMeter
from math import ceil
import models.cl_utils as cl_utils

logger = logging.get_logger(__name__)


def save_gcam(preds, labels, idx, vids, gcams, cfg, save_dir = 'gcam_samples'):
    """
    preds, labels, idx : [B]
    vids : [B, 3, T, H, W]
    gcams : [B, T, H, W]
    """
    #save_dir = 'gcam_samples'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    num_vids = preds.shape[0]
    for i in range(num_vids):
        v = vids[i].permute(1,2,3,0)
        l = cfg.TASK.TASK_SO_FAR[labels[i]]
        p = cfg.TASK.TASK_SO_FAR[preds[i][0]]
        l = cfg.TASK.CLASS_LIST[l]
        p = cfg.TASK.CLASS_LIST[p]
        gcam = gcams[i]
        v_idx = str(idx[i].numpy())
        num_frames = v.shape[0]

        save_dir_vid = os.path.join(save_dir, v_idx)
        if not os.path.exists(save_dir_vid):
            os.mkdir(save_dir_vid)
        for j in range(num_frames):
            frame = v[j].numpy() * cfg.DATA.STD + cfg.DATA.MEAN
            frame *= 255.0
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #gcam_frame = cm.jet_r(gcam[j].numpy())[..., :3] * 255.0
            gcam_frame = (gcam[j].numpy() * 255.0).astype(np.uint8)
            gcam_frame = cv2.applyColorMap(gcam_frame, cv2.COLORMAP_VIRIDIS)
            #save_image = 0.3 * gcam_frame.astype(np.float) + 0.7 * frame.astype(np.float)
            save_image = cv2.addWeighted(gcam_frame, 0.5, frame, 0.5, 0)
            #cv2.rectangle(output, (0, 0), (100, 10), (0, 0, 0), -1)
            #cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            #        0.8, (255, 255, 255), 2)
            file_name = "frame_{}_label_{}_pred_{}.png".format(
                    str(j), str(l), str(p))
            file_name = os.path.join(save_dir_vid,file_name)
            #print(file_name)
            cv2.imwrite(file_name, np.uint8(save_image))






@torch.no_grad()
def perform_visualization(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.train()
    #model.eval()
    test_meter.iter_tic()
    #current_age = torch.tensor([cfg.TASK.AGE]).cuda(non_blocking=True)

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        model.module.set_gradcam_hook()
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach().cpu(),
                ori_boxes.detach().cpu(),
                metadata.detach().cpu(),
            )
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            logger.info(model.module)
            preds = model(inputs)
            print(preds.size())
            aaa
            preds.requires_grad = True
            print(preds)
            _, top_idx = preds.sort(dim=1, descending=True)
            print(top_idx)
            gcam = cl_utils.grad_cam(model, top_idx, preds, preds.shape[1])
            gcam = gcam.view(-1, 1, 4, 8, 8) # We already know...
            B, C, T, H, W = inputs.shape
            #B_, C_, T_, H_, W_ = gcam.shape
            scale_factor = [T/4, H/7, W/7]
            upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
            gcam_upsampled = upsample(gcam)


            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx, inputs, gcam_upsampled = du.all_gather(
                    [preds, labels, video_idx, inputs, gcam_upsampled]
                )

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach().cpu(),
                labels.detach().cpu(),
                video_idx.detach().cpu(),
            )
            #test_meter.log_iter_stats(cur_iter)
            save_gcam(preds[:,top_idx].detach().cpu(), labels.detach().cpu(),
                    video_idx.detach().cpu(), inputs.detach().cpu(),
                    gcam_upsampled.detach.cpu())
            aaa
        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    test_meter.finalize_metrics()
    test_meter.reset()


def visualize(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Set up continual learning environment
    num_classes = cfg.MODEL.NUM_CLASSES
    num_task = cfg.TASK.NUM_TASK
    classes_per_task = ceil(num_classes/num_task)
    total_task_list = cfg.TASK.TASK_LIST

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    #for i in range(num_task):
    #for i in range(6): # For test
        #logger.info("----AGE {} VIS----".format(i))

    age = cfg.TASK.AGE

    cfg.TASK.CURRENT_HEAD = (age+1)*classes_per_task
    model = build_model(cfg)

    last_checkpoint = cu.get_age_checkpoint(cfg.OUTPUT_DIR, cfg.TASK.AGE)
    cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)

    if cfg.VISUALIZE.INCREMENT:
        age += 1
        cfg.TASK.CURRENT_HEAD = (age+1)*classes_per_task
        increment_model(mode, cfg)
    logger.info("Visualize Grad-CAM of {}-th DATA from {}-th Model".format(age, cfg.TASK.AGE))
    logger.info(model.module.head)
    cfg.TASK.CURRENT_TASK = total_task_list[:(age+1)*classes_per_task] # if we only consider the average accuracy
    cfg.TASK.TASK_SO_FAR = total_task_list[:(age+1)*classes_per_task]

    logger.info(cfg.TASK.CURRENT_TASK)
    #optimizer = optim.construct_optimizer(model, cfg)
    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.TASK.CURRENT_HEAD, #cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.TASK.CURRENT_TASK,
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Perform multi-view test on the entire dataset.
    #for p in model.parameters():
    #    p.requires_grad = True
    perform_visualization(test_loader, model, test_meter, cfg)
