import torch
import numpy
import os
from ops.models import TSN
from ops.utils import save_checkpoint

def weight_align(args, age, current_task, current_head):
    # Construct TSM Models
    model = TSN(num_class=current_head, num_segments=args.num_segments, modality='RGB',
            base_model=args.arch, consensus_type=args.consensus_type, dropout=args.dropout,
            img_feature_dim=args.img_feature_dim, partial_bn=not args.no_partialbn,
            pretrain=args.pretrain, is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
            fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
            temporal_pool=args.temporal_pool, non_local=args.non_local, cl_method=args.cl_method, num_proxy=args.num_proxy,
            age=age, cur_task_size=len(current_task))

    print("Load the Model")
    ckpt_path = os.path.join(args.root_model, args.dataset, str(args.nb_class), '{:03d}'.format(args.exp), 'task_{:03d}.pth.tar'.format(age))
    sd = torch.load(ckpt_path)
    sd = sd['state_dict']
    state_dict = dict()
    for k, v in sd.items():
        state_dict[k[7:]] = v

    model.load_state_dict(state_dict)

    model.weight_alignment(len(current_task))

    #state_dict['state_dict'] = model.state_dict(),
    new_state_dict = dict()
    for k, v in model.state_dict().items():
        new_k = 'module.'+k
        new_state_dict[new_k] = v
    state_dict['state_dict'] = new_state_dict

    save_checkpoint(args, age, state_dict)

