import torch
import numpy as np
from collections import OrderedDict
from ops.transforms import *
from ops.models import TSN
from ops.dataset import TSNDataSet, TSNDataSetBoth
from torch.utils.data import DataLoader
from ops.utils import *

def _remove_feat(features, idx, args):
    results = torch.cat((features[:idx,...],features[idx+1:,...]))
    return results


def _construct_exemplar_set(dloader, model, current_task, class_indexer, memory_size, args):
    model.eval()
    ex_dict = {}
    counter  = 0
    for i in current_task:
        ex_dict[class_indexer[i]] = {}

    with torch.no_grad():
        for i, (input, target, props) in enumerate(dloader):
            input = input.cuda()
            target = target
            # compute output
            outputs = model(input=input, only_feat=True, \
                    exemplar_selection=True if args.store_frames=='clips' else False)
            logits = outputs['preds']
            feat = outputs['feat']

            if args.store_frames == 'clips':
                #print(props)
                #print(feat.size())
                if args.clip_herding:
                    feat = feat.mean(2)
                elif args.frame_herding:
                    feat = feat.view(-1,args.exemplar_segments*args.num_segments,feat.size()[-1])
                    logits = logits.view(-1,args.exemplar_segments*args.num_segments,logits.size()[-1])
                    target_segs = target.repeat(args.exemplar_segments*args.num_segments,1).T
                    gt_index = torch.zeros(logits.size())
                    gt_index = gt_index.scatter(2,target_segs.view(-1,args.exemplar_segments*args.num_segments,1),1).ge(0.5).cuda()
                    gt_scores = logits.masked_select(gt_index)
                    gt_scores = gt_scores.view(-1,args.exemplar_segments*args.num_segments)
                    #highest_seg_index = torch.argmax(gt_scores,dim=1)

                    highest_seg_index = torch.topk(gt_scores,args.num_segments,dim=1)[1]
                    highest_seg_index = torch.sort(highest_seg_index)[0]
                    feat = feat.gather(1,highest_seg_index.unsqueeze(2).expand(feat.size()[0],args.num_segments,feat.size()[-1]))#[torch.arange(feat.size(0)),highest_seg_index]
                    feat = feat.mean(1)
                else:
                    logits = logits.view(-1,args.exemplar_segments,args.num_segments,logits.size()[-1])
                    #print(logits.size())
                    logits = logits.mean(2)
                    target_segs = target.repeat(args.exemplar_segments,1).T
                    gt_index = torch.zeros(logits.size())
                    gt_index = gt_index.scatter(2,target_segs.view(-1,args.exemplar_segments,1),1).ge(0.5).cuda()
                    gt_scores = logits.masked_select(gt_index)
                    gt_scores = gt_scores.view(-1,args.exemplar_segments)
                    #print(logits)
                    #print(gt_scores)

                    if args.rc:
                        highest_seg_index = torch.randint(args.exemplar_segments,(target.size(0),))
                    else:
                        highest_seg_index = torch.argmax(gt_scores,dim=1)
                    #print(highest_seg_index)
                    #print(props[-1])
                    feat = feat.mean(2)

                    feat = feat[torch.arange(feat.size(0)),highest_seg_index]
                    highest_seg_index = highest_seg_index * args.num_segments

            elif args.texemplar:
                feat = feat
            else:
                feat = feat.mean(1)
                #vindex = props[2]

            for j in range(target.size(0)):
                k = props[0][j]
                vn = props[1][j]
                #v = feat[j]
                v = []
                vind = []
                if args.store_frames == 'clips':
                    if args.clip_herding:
                        for m in range(args.exemplar_segments):
                            v.append(feat[j][m])
                            vind.append(props[2][j][m*args.exemplar_segments:(m+1)*args.exemplar_segments])
                    elif args.frame_herding:
                        v.append(feat[j])
                        vind.append(props[2][j][highest_seg_index[j]])
                    else:
                        v.append(feat[j])
                        vind.append(props[2][j][highest_seg_index[j]:highest_seg_index[j]+args.num_segments])
                else:
                    v.append(feat[j])
                    vind.append(props[2][j])
                #print(vind)
                if int(target[j]) in ex_dict.keys():
                    for m in range(len(v)):
                        ex_dict[int(target[j])].update({counter:(k,v[m],vn,vind[m])})
                        counter +=1

    exemplar_list = []

    for i in current_task:
        temp_dict = ex_dict[class_indexer[i]]
        paths = []
        features = []
        nframes = []
        inds = []

        for k, v in enumerate(temp_dict.items()):
            f_path = v[1][0]
            feat = v[1][1]
            feat = feat / torch.norm(feat,p=2)
            nframe = v[1][2]
            frame_ind = torch.unique(v[1][3],sorted=True)
            paths.append(f_path)
            features.append(feat)
            nframes.append(nframe)
            inds.append(frame_ind)

        features = torch.stack(features)
        class_mean = torch.mean(features,0)# axis=0)
        class_mean = class_mean / torch.norm(class_mean,p=2)

        exemplar_i = {}
        #nframe_i = []

        step_t = 0
        mu = class_mean
        w_t = mu

        while True:
            if len(exemplar_i.keys())>=memory_size: # or (step_t>=1.1*m) :
                break
            #elif sum(nframe_i)>m+1:
            #    break
            if features.size(0) == 0:
                break
            if args.texemplar:
                tmp_t = torch.einsum('bij,ij->bi',features,mu).sum(1)
            else:
                tmp_t = torch.matmul(features,mu) # dot w_t, features
            index = torch.argmax(tmp_t)
            w_t = w_t + mu - features[index]
            step_t += 1
            #print(paths[index])
            if paths[index] not in exemplar_i.keys():
                if args.store_frames=='entire':
                    exemplar_i[paths[index]] = (step_t,nframes[index],inds[index])
                    #nframe_i.append(nframes[index])
                else:
                    exemplar_i[paths[index]] = (step_t,len(inds[index]),inds[index])
                    #nframe_i.append(len(inds[index]))

            features = _remove_feat(features,index,args)
            del paths[index]
            del nframes[index]
            del inds[index]

        exemplar_i = OrderedDict(sorted(exemplar_i.items(), key=lambda x: x[1][0]))
        exemplar_list.append(exemplar_i)

    return exemplar_list

def _reduce_exemplar_set(ex_list, memory_size, store_frames): #budget_unit):
    reduced_list = []

    for i in range(len(ex_list)):
        ex_i = list(ex_list[i].items())
        #if budget_unit == 'video':
        #if store_frames == 'entire':
        reduced_list.append(OrderedDict(ex_i[:memory_size]))

    return reduced_list


def manage_exemplar_set(args, age, current_task, current_head, class_indexer, prefix):
    model = TSN(args, num_class=current_head, modality='RGB',
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                age=age,cur_task_size=len(current_task),training=True,fine_tune=True)

    print("Construct Exemplar Set")
    if args.budget_type == 'fixed':
        exemplar_per_class = args.K//current_head
    else:
        exemplar_per_class = args.K

    if age > 0:
        exemplar_dict = load_exemplars(args)#, current_head-len(current_task), age-1)
        exemplar_list = exemplar_dict[age-1]
    else:
        exemplar_dict = {}
        exemplar_list = None

    scale_size = model.scale_size
    input_size = model.input_size

    normalize = GroupNormalize(model.input_mean, model.input_std)

    print("Load the Model")
    ckpt_path = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp), 'task_{:03d}.pth.tar'.format(age))
    sd = torch.load(ckpt_path)
    sd = sd['state_dict']
    state_dict = dict()
    for k, v in sd.items():
        state_dict[k[7:]] = v

    model.load_state_dict(state_dict)

    print(model.new_fc)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    print("Exemplar per class : {}".format(exemplar_per_class))
    # Construct Exemplar Set for the Current Task
    transform_ex = torchvision.transforms.Compose([
                                            GroupScale(scale_size),
                                            GroupCenterCrop(input_size),
                                            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                            normalize,
                                            ])

    train_dataset_for_exemplar = TSNDataSet(args.root_path, args.train_list[0], current_task, class_indexer,
                            num_segments=args.num_segments, random_shift=False, new_length=1, #####
                            modality='RGB',image_tmpl=prefix[0], transform=transform_ex,
                            dense_sample=args.dense_sample, 
                            store_frames=args.store_frames, exemplar_segments=args.exemplar_segments) 


    train_loader_for_exemplar = DataLoader(train_dataset_for_exemplar, batch_size=args.exemplar_batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False)

    current_task_exemplar = _construct_exemplar_set(train_loader_for_exemplar,model,current_task,class_indexer,exemplar_per_class,args)

    if age > 0:
        # Reduce Exemplar Set
        if args.budget_type == 'fixed':
            exemplar_list = _reduce_exemplar_set(exemplar_list, exemplar_per_class, args.store_frames)
        exemplar_list = exemplar_list + current_task_exemplar
    else:
        exemplar_list = current_task_exemplar

    exemplar_dict[age] = exemplar_list
    save_exemplars(args, exemplar_dict)

