from utils.hsnet.model.learner import HPNLearner
from utils.hsnet.model.base.correlation import Correlation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import datetime
import time
import os
import math

from ops.models import TSN
from ops.dataset import TSNDataSet, TSNDataSetBoth
from ops.transforms import *
from ops.utils import *
import cl_methods.cl_utils as cl_utils
import cl_methods.distillation as cl_dist
import cl_methods.classifer as classifier
import cl_methods.regularizer as cl_reg
from cl_methods import tcd
import copy
import wandb

import utils.hsnet.model.hsnet as hsnet
import utils.hsnet.model.base.feature as hs_feature

from functools import reduce
from operator import add


def _train(args, train_loader, model, criterion, optimizer, epoch, age, lambda_0=[0.5,0.5], model_old=None, importance_list=None, regularizer=None):
    """
    Train the model only with RGB
    = Not using Flow
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_kd_logit = AverageMeter()
    losses_att = AverageMeter()
    losses_mr = AverageMeter()
    losses_div = AverageMeter()
    losses_reg = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()

    loss = torch.tensor(0.).cuda()
    loss_kd_logit = torch.tensor(0.).cuda()
    loss_att = torch.tensor(0.).cuda()
    loss_mr = torch.tensor(0.).cuda()
    loss_div = torch.tensor(0.).cuda()
    loss_reg = torch.tensor(0.).cuda()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    old_feats = []
    if model_old:
        model_old.eval()
        # # 提取旧类特征
        # old_feats = model_old.module.base_model.extract_feat_res(input.view((-1, 3) + input.size()[-2:]))


    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        # print(input.shape)
        target = target.cuda()

        # compute output
        #preds, feat, int_features = model(input, tsm=False)
        outputs = model(input=input,t_div=args.t_div)

        # 提取新类特征
        with torch.no_grad():
            new_feats = model.module.base_model.extract_feat_res(input.view((-1, 3) + input.size()[-2:]))

        # print(input.shape)
        preds = outputs['preds']
        feat = outputs['feat']
        int_features = outputs['int_features']
        importanceList = [model.module.base_model.importance1.importance,
                            model.module.base_model.importance2.importance,
                            model.module.base_model.importance3.importance,
                            model.module.base_model.importance4.importance,
                            model.module.base_model.importance_raw.importance]

        if args.loss_type == 'bce':
            target_ = cl_utils.convert_to_one_hot(target, preds.size(1))
        else:
            target_ = target

        if age > 0:
            if args.cl_type == 'DIST':
                #print(model.module.activations.keys())
                with torch.no_grad():
                    #for p in model_old.module.new_fc.parameters():
                    #    p.require_grad = True
                    outputs_old = model_old(input=input)

                    # 提取旧类特征
                    old_feats = model_old.module.base_model.extract_feat_res(input.view((-1, 3) + input.size()[-2:]))

                    preds_old = outputs_old['preds']
                    feat_old = outputs_old['feat']
                    int_features_old = outputs_old['int_features']
                    #print(model_old.module.activations.keys())
                    #aaa
                old_size = preds_old.size(1)
                preds_base = torch.sigmoid(preds_old)

                if args.cl_method in ['LWFMC','iCaRL','LWM']:
                    # LwF-MC, iCaRL
                    if args.loss_type == 'nll':
                        loss_ce = criterion(preds, target_)
                    elif args.loss_type == 'bce':
                        target_ = target_[..., old_size:]
                        loss_ce = criterion(preds[...,old_size:],target_)
                    loss_kd_logit = F.binary_cross_entropy_with_logits(preds[..., :old_size]/args.T, preds_base.data/args.T)
                    loss = lambda_0[0] * loss_ce + lambda_0[1] * loss_kd_logit

                    # LwM
                    if args.cl_method == 'LWM':
                        loss_att = cl_dist.gcam_dist(model, model_old, preds, int_features[-1],
                                int_features_old[-1], old_size, input.shape, args)
                        loss = loss + args.lambda_1 * loss_att

                # LUCIR
                elif args.cl_method == 'LUCIR':
                    loss_ce = criterion(preds, target_)
                    loss_kd_logit = cl_dist.lf_dist(feat,feat_old) #,args.num_segments)
                    loss_mr = cl_dist.mr_loss(preds,target_,old_size,args)
                    loss = lambda_0[0] * loss_ce + lambda_0[1] * loss_kd_logit + args.lambda_1 * loss_mr

                # POD-Net
                elif args.cl_method == 'POD':
                    loss_ce = cl_dist.nca_loss(preds, target_)
                    loss_kd_logit = cl_dist.lf_dist(feat,feat_old) #,args.num_segments)
                    loss_att = cl_dist.feat_dist(int_features,int_features_old,args)
                    loss = lambda_0[0] * loss_ce + lambda_0[1] * loss_kd_logit + (lambda_0[1]/args.lambda_0)*args.lambda_1 * loss_att

                elif args.cl_method == 'OURS':
                    if args.fc == 'lsc':
                        loss_ce = cl_dist.nca_loss(preds, target_)
                    else:
                        loss_ce = criterion(preds, target_)
                    # loss_att 是时间通道重要性loss，loss_ce是正常的loss，loss_kd_logit是，loss_div是正交性loss
                    # importance_list[0] = importance_list[0] * model.base_model.learneable.learnable1
                    # importance_list[1] = importance_list[1] * model.base_model.learneable.learnable2
                    # importance_list[2] = importance_list[2] * model.base_model.learneable.learnable3
                    # importance_list[3] = importance_list[3] * model.base_model.learneable.learnable4

                    #loss_kd_logit = cl_dist.lf_dist_tcd(feat,feat_old,factor=importance_list[-1] if importance_list else None)
                    #loss_att = cl_dist.feat_dist(int_features,int_features_old,args,factor=importance_list[:-1] if importance_list else None)
                    #loss = lambda_0[0] * loss_ce + lambda_0[1] * loss_kd_logit + args.lambda_1 * loss_att


                    #importanceList = np.array(importanceList)
                    #print("importanceList", importanceList)

                    # '''Get features'''
                    # nbottlenecks = [3, 4, 6, 3]
                    # feat_ids = list(range(4, 17))
                    # bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
                    # lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
                    # old_features_extracted = hs_feature.extract_feat_res(input, model_old, feat_ids, bottleneck_ids, lids)
                    # new_features_extracted = hs_feature.extract_feat_res(input, model, feat_ids, bottleneck_ids, lids)

                    # '''Constract hsnet'''
                    # hs_model = hsnet.HypercorrSqueezeNetwork()
                    
                    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # criterion = nn.CrossEntropyLoss().to(device)
                    # loss_MAS = mas_regularize.get_regularized_loss(criterion, outputs, target, model, 300000)
                    
                    # '''Constract hsnet'''
                    # hs_model = hsnet.HypercorrSqueezeNetwork()\

                    #引入hsnet
                    hs_model = hsnet.HypercorrSqueezeNetwork()
                    

                    with torch.no_grad():
                        nbottlenecks = [3,4,6,3]
                        lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
                        stack_ids = torch.tensor(lids).bincount().__reversed__().cumsum(dim=0)[:3]
                        print(stack_ids)
                        corr = Correlation.multilayer_correlation(old_feats, new_feats, stack_ids)
                        # print(corr)
                        hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
                        logit_mask = hpn_learner(corr)
                        print(logit_mask)

                    loss_kd_logit = cl_dist.lf_dist_tcd(feat,feat_old,factor=importanceList[-1] if importanceList else None)
                    loss_att = cl_dist.feat_dist(int_features,int_features_old,args,factor=importanceList[:-1] if importanceList else None)
                    loss = lambda_0[0] * loss_ce + lambda_0[1] * loss_kd_logit + args.lambda_1 * loss_att

                del preds_old, feat_old, int_features_old,old_feats,new_feats
                del feat, int_features

            else:
                loss_ce = criterion(preds, target_)
                loss = loss_ce
        else:
            #print(outputs['old_importance'][-1])
            if args.fc == 'lsc':
                loss_ce = cl_dist.nca_loss(preds, target_)#, margin=args.margin, \
                #        scale=model.module.new_fc.eta)
            else:
                loss_ce = criterion(preds, target_)
            loss = loss_ce
    
        if args.t_div:
            loss_div = outputs['t_div'].sum()/input.size(0)
            #loss_div = cl_dist.temporal_diversity_loss(feat, int_features, args.num_segments, args.t_div_ratio)
            loss = loss + args.lambda_2 * loss_div

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.cl_type == 'REG':
            regularizer.update()
            loss_reg = args.lambda_0 * regularizer.penalty()
            if age > 0:
                loss_reg.backward()

        if loss_reg != 0:
            loss += loss_reg

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(preds.data, target, topk=(1, 5))
        prec1 = accuracy(preds.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        losses_ce.update(loss_ce.item(), input.size(0))
        losses_kd_logit.update(loss_kd_logit.item(),input.size(0))
        losses_att.update(loss_att.item(), input.size(0))
        losses_mr.update(loss_mr.item(),input.size(0))
        losses_div.update(loss_div.item(),input.size(0))
        losses_reg.update(loss_reg.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))
        #top5.update(prec5.item(), input.size(0))

        del preds #, feat, int_features

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #printGPUInfo()

        if i % args.print_freq == 0:
            print(datetime.datetime.now())
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss CE {loss_ce.val:.4f} ({loss_ce.avg:.4f})\t'
                      'Loss KD (Logit) {loss_kd_logit.val:.4f} ({loss_kd_logit.avg:.4f})\t'
                      'Loss KD (GCAM) {loss_att.val:.4f} ({loss_att.avg:.4f})\t'
                      'Loss MR {loss_mr.val:.4f} ({loss_mr.avg:.4f})\t'
                      'Loss DIV {loss_div.val:.4f} ({loss_div.avg:.4f})\t'
                      'Loss REG {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_ce=losses_ce, loss_kd_logit=losses_kd_logit,
                loss_att=losses_att, loss_mr=losses_mr, loss_div=losses_div, loss_reg=losses_reg,
                top1=top1, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            #print("importanceList",importanceList)
            if args.wandb:
                wandb.log({"task_{}/loss".format(age): losses.avg,
                    "task_{}/loss_ce".format(age): losses_ce.avg,
                    "task_{}/loss_kd (logit)".format(age): losses_kd_logit.avg,
                    "task_{}/loss_kd (att)".format(age): losses_att.avg,
                    "task_{}/loss_mr (att)".format(age): losses_mr.avg,
                    "task_{}/loss_div".format(age): losses_div.avg,
                    "task_{}/loss_reg".format(age): losses_reg.avg,
                    "task_{}/train_top1".format(age): top1.avg,
                    #"task_{}/train_top5".format(age): top5.avg,
                    "task_{}/lr".format(age): optimizer.param_groups[-1]['lr'] * 0.1
                    })

        #torch.cuda.empty_cache()


def _validate(args, val_loader, model, criterion, age):
    batch_time = AverageMeter()
    #losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            outputs = model(input=input)
            output = outputs['preds']

            target_one_hot = cl_utils.convert_to_one_hot(target, output.size(1))

            #loss = criterion(output, target_one_hot)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            prec1 = accuracy(output.data, target, topk=(1,))

            #losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))
            #top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          #'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, #loss=losses,
                    top1=top1))
                #print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} '#Loss {loss.avg:.5f}'
              .format(top1=top1))#, loss=losses))
    print(output)
    if args.wandb:
        wandb.log({"task_{}/val_top1".format(age): top1.avg,
        #"task_{}/val_top5".format(age): top5.avg
        })

    return top1.avg

def train_task(args, age, current_task, current_head, class_indexer, model_old=None, model_flow=None, prefix=None):
    #K = args.K
    #current_task = total_task[age]
    if age == 1 and args.dataset=='hmdb51' and args.fc=='cc':
        args.lr = args.lr * 0.1
    hook = False
    if age > 0 and args.cl_type=='DIST':
        hook = True
    if age > 0 and args.exemplar:
        exemplar_dict = load_exemplars(args)#, current_head-len(current_task), age-1)
        exemplar_list = exemplar_dict[age-1]
        exemplar_per_class = len(exemplar_list[0])
    else:
        #exemplar_dict = {}
        exemplar_list = None
        exemplar_per_class = 0
        #exemplar_nframe_list = None
    if age > 0 and args.use_importance:
        importance_dict = load_importance(args)
        importance_temp = importance_dict[age-1]
        importance_list = []
        for i in importance_temp:
            if args.importance_project:
                i = cl_utils.project_importance(i)
            importance_list.append(i)

    else:
        importance_dict = {}
        importance_list = None

    regularizer = None
    old_state = None

    # Construct TSM Models
    model = TSN(args, num_class=current_head if age==0 else current_head-len(current_task), modality='RGB',
            fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
            apply_hooks=hook, age=age, training=True,
            cur_task_size=len(current_task),exemplar_segments=args.exemplar_segments)

    #print(model.state_dict().keys())
    #aaaa
    scale_size = model.scale_size
    input_size = model.input_size

    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)
    transform_rgb = torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])
    lambda_0 = 0.0

    if age > 0:
        #elif args.cl_type == 'REG':
        print("Load the Previous Model")
        ckpt_path = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp), 'task_{:03d}.pth.tar'.format(age-1))
        sd = torch.load(ckpt_path)
        if 'old_state' in sd:
            old_state = sd['old_state']
        sd = sd['state_dict']
        state_dict = dict()
        for k, v in sd.items():
            state_dict[k[7:]] = v

        model.load_state_dict(state_dict)

        # Prepare Old Model to Distill
        if args.cl_type == 'DIST' or args.cl_type == 'REG':
            print("Copy the old Model")
            model_old = copy.deepcopy(model)
            model_old.eval()
            lambda_0 = cl_utils.set_lambda_0(len(current_task),current_head-len(current_task),args)
            #lambda_0 = (current_head - len(current_task)) / current_head
            print('lambda_0  : {}'.format(lambda_0))

        # Increment the classifier 
        print("Increment the Model")
        model.increment_head(current_head,age)
    print(model.new_fc)

    # Construct DataLoader
    if args.cl_type == 'REG':
        regularizer = cl_reg.get_regularizer(model, model_old, args.cl_method, current_head, args.nb_class, args.ewc_alpha,  old_state)
    else:
        regularizer = None

    if age > 0 and args.bic:
        if args.budget_type == 'fixed':
            exemplar_per_class = args.K//(current_head-len(current_task))
        else:
            exemplar_per_class = args.K
    train_dataset = TSNDataSet(args.root_path, args.train_list[0], current_task, class_indexer, num_segments=args.num_segments,
                new_length=1, modality='RGB',image_tmpl=prefix[0], transform=transform_rgb, dense_sample=args.dense_sample,
                exemplar_list=exemplar_list, is_entire=(args.store_frames=='entire'),
                bic='train' if args.bic else None,
                nb_val=args.nb_val,exemplar_per_class=exemplar_per_class, current_head=current_head,
                diverse_rate=args.diverse_rate, sample_weight=args.weight_sampling)#,budget_unit=args.budget_unit)
    '''
    val_dataset = TSNDataSet(args.root_path, args.val_list[0], current_task, class_indexer, num_segments=args.num_segments, random_shift=False,
                new_length=1, modality='RGB',image_tmpl=prefix[0], transform=transform_rgb, dense_sample=args.dense_sample)
    '''

    if age > 0 and args.weight_sampling:
        sampler = torch.utils.data.WeightedRandomSampler(weights=train_dataset.sample_weights,
                                                        num_samples=len(train_dataset.sample_weights),
                                                        replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                            sampler=sampler, num_workers=args.workers,
                            pin_memory=True, drop_last=True)

    else:
        #sampler = None
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=args.workers,
                            pin_memory=True, drop_last=True)
    '''
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=True)
    '''
    #print("DataLoader Constructed : Train {} / Val {}".format(len(train_loader), len(val_loader)))
    print("DataLoader Constructed : Train {}".format(len(train_loader)))

    policies = model.get_optim_policies()

    # Wrap the model with DataParallel module
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if model_old:
        model_old = torch.nn.DataParallel(model_old, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.loss_type == 'nll':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss().cuda()

    print("Optimizer Constructed")

    if age > 0 and args.fc in ['cc','lsc']: #(args.cl_method == 'LUCIR' or args.cl_method == 'POD'):
        transform_ex = torchvision.transforms.Compose([
                                                GroupScale(scale_size),
                                                GroupCenterCrop(input_size),
                                                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                                normalize,
                                                ])

        dataset_for_embedding = TSNDataSet(args.root_path, args.train_list[0], current_task, class_indexer,
                                num_segments=args.num_segments, random_shift=False, new_length=1,
                                modality='RGB',image_tmpl=prefix[0], transform=transform_ex,
                                dense_sample=args.dense_sample) #budget_unit=args.budget_unit,
        #                        store_frames=None)


        loader_for_embedding = DataLoader(dataset_for_embedding, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False)


        cl_utils.init_cosine_classifier(model,current_task,class_indexer,loader_for_embedding,args)

    best_prec1 = 0
    best_epoch = 0
    # Train the model for the current task
    for epoch in range(args.start_epoch, args.epochs):
        _adjust_learning_rate(args, optimizer, epoch, args.lr_type, args.lr_steps)
        #print("Learning rate adjusted")
        if args.use_flow:
            # Use flow as auxiliary
            _train_aux(args, train_loader, model, criterion, optimizer, epoch, model_old, model_flow)
        else:
            # RGB Only
            _train(args, train_loader, model, criterion, optimizer, epoch, age, regularizer=regularizer, lambda_0=lambda_0, model_old=model_old, importance_list=importance_list)
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            '''
            prec1 = _validate(args, val_loader, model, criterion, age)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            print('Best Prec@1 for the CURRENT task: {:.3f}'.format(best_prec1))
            '''
            #if args.cl_method in ['LUCIR','POD']:
            if args.fc in ['cc','lsc']:
                print('Sigma : {}, Eta : {}'.format(model.module.new_fc.sigma, model.module.new_fc.eta))

            state = {'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'old_state': regularizer.state_dict() if regularizer else None,
                'best_prec1': best_prec1,}
            save_checkpoint(args, age, state)#, is_best)

    if args.use_importance:
        tcd.compute_importance(args,model,train_loader,criterion)
        save_importance(args,age,model,importance_dict)

    del model
    #return model

def _adjust_learning_rate(args, optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


