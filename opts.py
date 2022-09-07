# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--exp', type=int, default=0)
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--modality', type=str, default='Both', choices=['RGB', 'Flow', 'Both'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet34") #default="BNInception")
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.8, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll', 'bce'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')
parser.add_argument('--test_crops', default=5, type=int)

# ========================= Learning Configs ==========================
parser.add_argument('--training', default=False, action='store_true')
parser.add_argument('--testing', default=False, action='store_true')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--train_batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (train) (default: 256)')
parser.add_argument('-b_test', '--test_batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (test) (default: 256)')
parser.add_argument('-b_ex', '--exemplar_batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (exemplar) (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[20, 30], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--mars_weight', '--mw', default=1e-3, type=float)

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--wandb', default=False, action="store_true", help='use wandb to log')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--shift', default=True, action="store_false", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--twice_sample', default=False, action="store_true")

# ========================= Continual Configs ==========================
parser.add_argument('--nb_class', default=10, type=int, help='class batch')
#parser.add_argument('--num_task', default=10, type=int, help='number of tasks')
parser.add_argument('--init_task', default=51, type=int, help='size of the initial task')
parser.add_argument('--start_task', default=0, type=int, help='starting task')
parser.add_argument('--end_task', default=100, type=int, help='last task')
parser.add_argument('--use_flow', default=False, action="store_true", help='use flow model')
parser.add_argument('--exemplar', default=False, action="store_true", help='use exemplars')
parser.add_argument('--cl_type',default='FT', type=str, choices=['FT', 'DIST', 'REG'])
parser.add_argument('--cl_method', default=None, type=str, choices=['LWFMC','iCaRL','LWM','LUCIR','POD','OURS','EWC','SI'])
parser.add_argument('--fc', default='linear', type=str, choices=['linear','cc','lsc'])
parser.add_argument('--T', default=1.0,type=float,help='Distillation Temperature')
parser.add_argument('--lambda_0', default=1.0,type=float, help='weight for Distillation')
parser.add_argument('--lambda_1', default=1e-3, type=float, help='weight for [Attention Distillation Loss (LWM), Margin Ranking Loss (LUCIR)]')
parser.add_argument('--lambda_2', default=1e-4,type=float,help='weight for temporal diversity loss')
parser.add_argument('--t_div',default=False,action="store_true",help='use temporal diversitiy loss')
parser.add_argument('--t_div_ratio', default=1.0,type=float,help='ratio of the channels for div_loss')
parser.add_argument('--K', default=20, type=int, help='memory budget for the exemplar set')
parser.add_argument('--nme', default=False, action="store_true", help='nme_classifier')
parser.add_argument('--budget_type',default='class', type=str, choices=['fixed','class'])
parser.add_argument('--store_frames',default=None, type=str, choices=['uniform','random','entire','clips'])
parser.add_argument('--lambda_0_type', default='fix', type=str, choices=['fix','arith','geo'])
parser.add_argument('--margin', default=0.5, type=float)
parser.add_argument('--mr_k', default=2, type=int)
parser.add_argument('--num_proxy',default=1,type=int,help='number of the proxies for cosine classifier (POD)')
parser.add_argument('--cbf', default=False, action="store_true", help='class balanced fine-tuning')
parser.add_argument('--fine_tune_epochs', default=20, type=int,
                    help='number of total cbf epochs to run')
parser.add_argument('--fine_tune_lr', default=1e-4, type=float,
                    help='lr of cbf')
parser.add_argument('--wa', default=False, action="store_true", help='Weight Alignment')
parser.add_argument('--exemplar_segments', default=8,type=int,help='exemplar candidate segments')
parser.add_argument('--rc', default=False, action="store_true", help='random clip')
parser.add_argument('--clip_herding', default=False, action="store_true", help='ours')
parser.add_argument('--frame_herding', default=False, action="store_true", help='ours2')
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--sigma_learnable', default=False, action="store_true")
parser.add_argument('--eta', default=1.0, type=float)
parser.add_argument('--eta_learnable', default=False, action="store_true")
parser.add_argument('--nca_margin',default=0.6,type=float)
parser.add_argument('--use_importance',default=False,action="store_true")
parser.add_argument('--importance', default='c', type=str,choices=['c','tc']) #action="store_true")
parser.add_argument('--importance_project',default=False,action="store_true")
parser.add_argument('--factorize', default='T-S',type=str,choices=['T-S','TH-TW','T-GAP','T-POD','T-H-W','spatial_pixel','pixel','channel','all'])
parser.add_argument('--tnme',default=False,action="store_true")
parser.add_argument('--texemplar',default=False,action="store_true")
parser.add_argument('--bic',default=False,action="store_true")
parser.add_argument('--nb_val',default=0.1,type=float)
parser.add_argument('--ewc_alpha', default=0.9,type=float,help='EWC Alpha')
parser.add_argument('--diverse_rate',default=False,action="store_true")
parser.add_argument('--weight_sampling', default=False, action="store_true")

