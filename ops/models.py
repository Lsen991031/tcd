# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn
import torch.nn.functional as F
import ops.resnet_models as models
from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import copy
import threading
from cl_methods.cosine_linear import CosineLinear, SplitCosineLinear

class TSN(nn.Module):
    def __init__(self, args, num_class, modality, new_length=None, before_softmax=True,
                 crop_num=1, print_spec=True, fc_lr5=False, apply_hooks=False,
                 age=0, training=False, cur_task_size=0, exemplar_segments=None, fine_tune=False):
        super(TSN, self).__init__()

        self.modality = modality
        self.num_class = num_class
        self.num_segments = args.num_segments
        self.exemplar_segments = exemplar_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = args.dropout
        self.crop_num = crop_num
        self.consensus_type = args.consensus_type
        self.pretrain = args.pretrain

        self.is_shift = args.shift
        self.shift_div = args.shift_div
        self.shift_place = args.shift_place
        self.base_model_name = args.arch #base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = args.temporal_pool
        self.non_local = args.non_local
        self.apply_hooks = apply_hooks
        self.remove_last_relu = False
        self.num_proxy = args.num_proxy
        self.age = age
        self.training = training
        self.fine_tune = fine_tune

        self.cl_method = args.cl_method
        self.sigma_learnable = args.sigma_learnable
        self.sigma = args.sigma
        self.eta_learnable = args.eta_learnable
        self.eta = args.eta
        self.nca_margin = args.nca_margin
        self.fc_type = args.fc

        self.reg_params = {}

        
        #if self.cl_method == 'LUCIR' or self.cl_method == 'POD':
        if self.fc_type in ['cc','lsc']:
            self.remove_last_relu = True

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        self.importance = args.importance
        '''
        if print_spec:
            print(("""
        Initializing TSN with base model: {}.
        TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))
        '''
        self._prepare_base_model(self.base_model_name)

        feature_dim = self._prepare_tsn(cur_task_size)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(self.consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = not args.no_partialbn
        if self._enable_pbn:
            self.partialBN(True)

        '''
        self._hooks = [None]
        self.activations = {}

        def forward_hook(module, input, output):
            _, C, H, W = output.shape
            act = output.view(-1, self.num_segments, C, H, W)
            self.activations[threading.get_ident()] = act


        #self._hooks[0] = self.base_model.layer4.register_backward_hook(backward_hook)
        self._hooks[0] = self.base_model.layer4.register_forward_hook(forward_hook)

        '''
        # Hooks
        if self.apply_hooks:
            self._hooks = [None,None]
            self.set_hooks()

    def _prepare_tsn(self, cur_task_size): # num_class = old_class for age > 0
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, self.num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.fc_type in ['cc','lsc']: #self.cl_method=='LUCIR' or self.cl_method=='POD':
                if self.age == 0:
                    self.new_fc = CosineLinear(feature_dim, self.num_class, self.num_proxy, self.sigma_learnable, self.sigma,
                                                self.eta_learnable, self.eta, version=self.fc_type, nca_margin=self.nca_margin,is_train=self.training)
                elif self.age == 1 and self.training and not self.fine_tune:
                    self.new_fc = CosineLinear(feature_dim, self.num_class, self.num_proxy, self.sigma_learnable, self.sigma,
                                                self.eta_learnable, self.eta, version=self.fc_type, nca_margin=self.nca_margin,is_train=self.training)
                elif self.training and not self.fine_tune: # Load old one to restore
                    self.new_fc = SplitCosineLinear(feature_dim, self.num_class-cur_task_size, cur_task_size,
                                                self.num_proxy, self.sigma_learnable, self.sigma,
                                                self.eta_learnable, self.eta, version=self.fc_type, nca_margin=self.nca_margin,is_train=self.training)
                else:
                    self.new_fc = SplitCosineLinear(feature_dim, self.num_class-cur_task_size, cur_task_size,
                                                self.num_proxy, self.sigma_learnable, self.sigma,
                                                self.eta_learnable, self.eta, version=self.fc_type, nca_margin=self.nca_margin,is_train=self.training)
            else:
                self.new_fc = nn.Linear(feature_dim, self.num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        elif self.fc_type == 'linear': #self.cl_method != 'LUCIR' and self.cl_method != 'POD': # CosineLinear has its own initializer
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(models, base_model)(True if self.pretrain == 'imagenet' else False,
                    self.remove_last_relu, self.importance, self.num_segments)
            if self.is_shift:
                #print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            #if self.importance is not None:

            if self.non_local:
                print('Adding non-local module...')
                #from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224 #224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        #if self.print_spec:
                        #    print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                #print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def increment_head(self, inc_class, age):
        #old_head = copy.deepcopy(self.new_fc)
        in_features = self.new_fc.in_features
        out_features = self.new_fc.out_features

        if self.fc_type in ['cc','lsc']: #self.cl_method == 'LUCIR' or self.cl_method == 'POD':
            new_head = SplitCosineLinear(in_features, out_features, inc_class-out_features, self.num_proxy,
                                        self.sigma_learnable, self.sigma,
                                        self.eta_learnable, self.eta, version=self.fc_type, nca_margin=self.nca_margin,is_train=self.training)

            if age == 1:
                new_head.fc1.weight.data = self.new_fc.weight.data.clone()
                if self.sigma_learnable:
                    new_head.sigma.data = self.new_fc.sigma.data.clone()
                if self.eta_learnable:
                    #print(self.new_fc.eta_learnable)
                    new_head.eta.data = self.new_fc.eta.data.clone()
            else:
                out_features1 = self.new_fc.fc1.out_features
                out_features2 = self.new_fc.fc2.out_features

                new_head.fc1.weight.data[:out_features1*self.num_proxy] = self.new_fc.fc1.weight.data.clone()
                new_head.fc1.weight.data[out_features1*self.num_proxy:] = self.new_fc.fc2.weight.data.clone()
                if self.sigma_learnable:
                    new_head.sigma.data = self.new_fc.sigma.data.clone()
                if self.eta_learnable:
                    new_head.eta.data = self.new_fc.eta.data.clone()
        else:
            new_head = nn.Linear(in_features, inc_class)
            if hasattr(new_head, 'weight'):
                    normal_(new_head.weight, 0, std=1e-3)
                    constant_(new_head.bias, 0)
            new_head.weight.data[:out_features] = self.new_fc.weight.data.clone()
            new_head.bias.data[:out_features] = self.new_fc.bias.data.clone()
        del self.new_fc
        self.new_fc = new_head

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            #print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            #print(m)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, CosineLinear): # or isinstance(m, pod.CosineLinear):
                ps = list(m.parameters())
                if self.age > 0 and ps[0].size()[0] == self.num_proxy * self.num_class:
                    #print(ps[0].size())
                    continue
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                    if len(ps) > 1:
                        for p in ps[1:]:
                            lr5_weight.append(p) # sigma, eta
                else:
                    normal_weight.append(ps[0])
                    if len(ps) > 1:
                        for p in ps[1:]:
                            normal_weight.append(p) # sigma, eta
            elif isinstance(m, SplitCosineLinear): # or isinstance(m, pod.SplitCosineLinear):
                ps = list(m.parameters())
                for p in ps:
                    #print(p.size())
                    if p.size()[0] == self.new_fc.fc1.out_features:
                        #print(p.size())
                        continue
                    if self.fc_lr5:
                        lr5_weight.append(p)
                    else:
                        normal_weight.append(p)

            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, models.Channel_Importance_Measure):
                continue
            #elif isinstance(m, models.Temporal_Channel_Importance_Measure):
            #    continue
            elif isinstance(m, models.Channel_Learnable_Importance):
                ps = list(m.parameters())
                for p in ps:
                    custom_ops.append(p)            
            elif isinstance(m, models.Learnable_Weight):
                ps = list(m.parameters())
                for p in ps:
                    custom_ops.append(p)
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def get_cbf_optim_policies(self):
        normal_weight = []
        #normal_bias = []
        lr5_weight = []

        #ignored_params = list(map(id, self.new_fc.parameters()))
        #base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        for m in self.modules():
            if isinstance(m, CosineLinear): # or isinstance(m, pod.CosineLinear):
                ps = list(m.parameters())
                if self.age > 0 and ps[0].size()[0] == self.num_proxy * self.num_class:
                    #print(ps[0].size())
                    continue
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                    if len(ps) > 1:
                        for p in ps[1:]:
                            lr5_weight.append(p) # sigma, eta
                else:
                    normal_weight.append(ps[0])
                    if len(ps) > 1:
                        for p in ps[1:]:
                            normal_weight.append(p) # sigma, eta
            elif isinstance(m, SplitCosineLinear): # or isinstance(m, pod.SplitCosineLinear):
                ps = list(m.parameters())
                for p in ps:
                    #print(p.size())
                    if p.size()[0] == self.new_fc.fc1.out_features:
                        #print(p.size())
                        continue
                    if self.fc_lr5:
                        lr5_weight.append(p)
                    else:
                        normal_weight.append(p)
        '''
        return [
                {'params':base_params, 'lr_mult':0, 'decay_mult':1, 'name':'base_params'},
                {'params':self.new_fc.parameters(), 'lr_mult':5, 'decay_mult':1, 'name':'fc'},
                {'params':self.new_fc.fc1.parameters(), 'lr_mult':5, 'decay_mult':1, 'name':'fc1'},
                {'params':self.new_fc.fc2.parameters(), 'lr_mult':5, 'decay_mult':1, 'name':'fc2'},
                ]
        '''
        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
        ]

    def forward(self, input, target=None, no_reshape=False, only_feat=False, exemplar_selection=False,t_div=False):
        outputs = {}

        # 提取旧类特征
        #feats = model_old.module.base_model.extract_feat_res(input.view((-1, 3) + input.size()[-2:]))



        # print(input.shape)
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out, int_features = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
            #print(input.shape)
            #print("***********")
        else:
            base_out, int_features = self.base_model(input)
        
        # print('1', base_out.shape)
        feature_conv5 = base_out
        if exemplar_selection:
            feature_conv5 = feature_conv5.view(-1,self.exemplar_segments,self.num_segments,feature_conv5.size()[-1])#.mean(1)
        else:
            feature_conv5 = feature_conv5.view(-1,self.num_segments,feature_conv5.size()[-1])#.mean(1)
        # print('2', feature_conv5.shape)

        if self.dropout > 0:
            if self.importance is not None:
                base_out = self.base_model.raw_features_importance(base_out)
                #importance.append(self.base_model.raw_features_importance.importance)
                #outputs['old_importance'] = importance
            base_out = base_out.view(-1,base_out.size()[-1])

            base_out = self.new_fc(base_out)
        
        if only_feat:
            outputs['preds'] = base_out
            outputs['feat'] = feature_conv5
            return outputs #nn.Softmax()(base_out)

        #feature_conv5 = feature_conv5.mean(1)
        if not self.before_softmax:
            base_out = self.softmax(base_out)

        #print('2', base_out.shape)
        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            #print('3', base_out.shape)
            output = self.consensus(base_out)

            outputs['preds'] = output.squeeze(1) # [8,51]
            # print('4', output.squeeze(1).shape)
            outputs['feat'] = feature_conv5 # [8,8,512]
            #print('feature_conv5.shape', feature_conv5.shape)
            outputs['int_features'] = int_features
            if t_div:
                outputs['t_div'] = self._get_div(int_features)
            # return output.squeeze(1), feature_conv5, int_features #layer4_out #, result_int_layer
            
            return outputs

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])


    def unset_hooks(self):
        self._hooks[0].remove()
        self._hooks[1].remove()
        self._hooks[0] = None
        self._hooks[1] = None
        #self._gradcam_gradients, self.activations = [None], [None]
        self._gradcam_gradients = [None]
        self.activations = {}


    def set_hooks(self):
        #self._gradcam_gradients, self.activations = [None], [None]
        self._gradcam_gradients = [None]
        self.activations = {}

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            #print('forward_hook')
            _, C, H, W = output.shape
            act = output.view(-1, self.num_segments, C, H, W)
            self.activations[threading.get_ident()] = act
            return None

        self._hooks[0] = self.base_model.layer4.register_backward_hook(backward_hook)
        self._hooks[1] = self.base_model.layer4.register_forward_hook(forward_hook)

    def _get_div(self, int_feat):
        num_layers = len(int_feat)
        num_pairs = self.num_segments * (self.num_segments-1)
        loss_div = torch.tensor(0.).cuda(non_blocking=True)
        for i in range(num_layers):
            f = int_feat[i]
            _, C, H, W = f.shape
            '''
            if t_div_ratio < 1.0:
                C = int(C * t_div_ratio)
                f = f[:,:C,:,:]
            '''
            f = f.view(-1, self.num_segments, C, H, W) # (B, T, C', H,W)
            f = f.view(-1, self.num_segments, C, H*W)  # (B, T, C', H*W)
            f_1 = f.permute(0,2,1,3) # (B, C', T, H*W)
            f_1_norm = f_1.norm(p=2, dim=3, keepdim=True)
            f_2 = f.permute(0,2,3,1) # (B, C', H*W, T)
            f_2_norm = f_2.norm(p=2, dim=2, keepdim=True)

            R = torch.matmul(f_1,f_2)/(torch.matmul(f_1_norm, f_2_norm) + 1e-6) # (B, C', T, T)
            I = torch.eye(self.num_segments).cuda()
            I = I.repeat((R.shape[0], R.shape[1], 1, 1))

            loss_i = F.relu(R-I).sum(-1).sum(-1) / num_pairs
            #print(loss_i.size())
            #loss_i = F.frobenius_norm(R-I)
            loss_i = loss_i.mean(-1).sum()
            #print(loss_i.size())

            loss_div = loss_div + loss_i

        loss_div = loss_div/num_layers
        return loss_div


    def weight_alignment(self,num_new_class):

        if isinstance(self.new_fc, nn.Linear):
            old_weight = self.new_fc.weight.data[:-num_new_class] # old, 2048
            new_weight = self.new_fc.weight.data[-num_new_class:] # new, 2048
        elif isinstance(self.new_fc, SplitCosineLinear): # or isinstance(self.new_fc, pod.SplitCosineLinear):
            old_weight = self.new_fc.fc1.weight.data
            new_weight = self.new_fc.fc2.weight.data
        else:
            raise NotImplementedError

        old_weight_norm = np.linalg.norm(old_weight,ord=2,axis=1)
        new_weight_norm = np.linalg.norm(new_weight,ord=2,axis=1)

        gamma = np.mean(old_weight_norm)/np.mean(new_weight_norm)
        print("Gamma : {}".format(gamma))

        updated_new_weight = gamma*new_weight

        if isinstance(self.new_fc, nn.Linear):
            self.new_fc.weight.data[-num_new_class:] = updated_new_weight
        elif isinstance(self.new_fc, SplitCosineLinear): # or isinstance(self.new_fc, pod.SplitCosineLinear):
            self.new_fc.fc2.weight.data = updated_new_weight

