import numpy as np
import random
import torch
import torch.nn.parallel
import torch.optim
import wandb
import os

from opts import parser
from ops import dataset_config, utils
from cl_methods import cl_utils
from cl_methods import exemplars
import train.train_i_cl as train_i_cl
import train.cbf as cbf
from train.calibration import bias_correction as bic
from cl_methods.weight_alignment import weight_align
from evaluation import eval_task

def main():
    # Load Args
    args = parser.parse_args()

    # Set Experiments
    if args.wandb:
        wandb.init(project='cilar2_{}_{:d}_{:d}'.format(args.dataset, args.init_task, args.nb_class), name="{:03d}".format(args.exp))
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,'Both')
    args.num_task = int(np.ceil((num_class - args.init_task)/args.nb_class)) + 1

    # Set Seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set Tasks
    class_list_total = np.arange(num_class)
    np.random.shuffle(class_list_total)
    total_task_list = class_list_total.tolist()
    class_indexer = dict((i, n) for n, i in enumerate(total_task_list)) # convert shuffled classes into 0,1,2,....
    utils.check_rootfolders(args)

    model_flow = None
    model_old = None
    bic_model = None

    total_task_list = cl_utils.set_task(args, total_task_list, num_class)
    current_head = 0

    

    end_task = min(args.end_task, args.num_task)

    for i in range(args.start_task, end_task):
        print('Method : {}'.format('FT' if args.cl_type=='FT' else args.cl_method))
        print("----AGE {}----".format(i))
        current_task = total_task_list[i]
        current_head = sum(len(j) for j in total_task_list[:i+1])
        print('current_task ', current_task)
        print('current_head ', current_head)

        # Phase 1 : Train Flow Model (Optional, Not now)
        if args.use_flow:
            print("Phase 1 : Train Flow Model")
            # train_i_only(args, i, modality='Flow', current_task, current_head, task_so_far, prefix=prefix)
            raise NotImplementedError
        # Phase 2 : Train RGB in an incremental manner
        print("Phase 2 : Train RGB Model in an Incremental Manner")
        if args.training:
            train_i_cl.train_task(args, i, current_task, current_head, class_indexer, model_flow=model_flow, prefix=prefix)
            if i > 0 and args.bic:
                bic_model = bic(args, i, total_task_list[:i+1], current_head, class_indexer, prefix)
        if args.exemplar:
            print("Phase 3 : Manage Exemplar Sets")
            exemplars.manage_exemplar_set(args, i, current_task, current_head, class_indexer, prefix=prefix)
        else:
            print("Phase 3 : Manage Exemplar Sets: SKIP (Does not use exemplar set)")
        # Phase 3 : Class-balanced Fine-tuning
        if i > 0 and args.cbf:
            print("Phase 4 : Class-balanced Fine-Tuning")
            cbf.train_task(args, i, total_task_list[:i+1], current_head, class_indexer, prefix)
        else:
            print("Phase 4 : Class-balanced Fine-Tuning : SKIP")
        if i > 0 and args.wa:
            print("Weight Alignment")
            weight_align(args, i, current_task, current_head)
        # Phase 4 : Test the Model for the tasks so far
        if args.testing:
            print("Phase 5 : Eval RGB Model for the Tasks Trained so far")
            n_test_vids = eval_task(args, i, total_task_list[:i+1], current_head, class_indexer,
                                len(current_task),prefix=prefix[0], bic_model=bic_model)
    
        torch.cuda.empty_cache()
    if args.testing:
        cl_utils.compute_final_stats(n_test_vids, args, 'cnn')
        if args.nme:
            cl_utils.compute_final_stats(n_test_vids, args, 'nme')

if __name__ == '__main__':
    main()
