import torch
import torch.nn as nn
import torch.nn.functional as F
import cl_methods.cl_utils as cl_utils
from math import sqrt

def kd_loss(p_old, p_now, temp=2.0):
    old_size = p_old.size(1)
    
    logp = F.log_softmax(p_now[:,:old_size]/temp, dim=1)
    logp_old = F.softmax(p_old.detach().clone()/temp, dim=1)

    outputs = torch.sum(logp*logp_old, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
 
    return outputs * temp * temp

def gcam_dist(model, model_old, p_now, feat, feat_old, old_size,
            input_shape, args):
    feat = feat.view(-1,args.num_segments,feat.size()[1],feat.size()[2],feat.size()[3])
    feat_old = feat_old.view(-1,args.num_segments,feat_old.size()[1],feat_old.size()[2],feat_old.size()[3])

    p_now_base = p_now[:, :old_size]
    _, top_base_idx = p_now_base.sort(dim=1, descending=True)
    top_base_ids = top_base_idx[:,[0]]

    gcam = cl_utils.grad_cam(model, feat, top_base_ids)
    gcam_old = cl_utils.grad_cam(model_old, feat_old, top_base_ids)
    #print(gcam.shape)
    B, _, Tm, Hm, Wm = gcam.shape

    gcam = gcam.view(B,-1) # B, Tm*Hm*Wm
    gcam = gcam/(gcam.norm(dim=1,keepdim=True) + 1e-8) # B, Tm*Hm*Wm
    gcam_old = gcam_old.view(B,-1) # B, Tm*Hm*Wm
    gcam_old = gcam_old/(gcam_old.norm(dim=1,keepdim=True) + 1e-8) # B, Tm*Hm*Wm

    #loss_att = torch.abs(gcam_old.clone().detach() - gcam).sum(dim=1).mean()
    loss_att = torch.abs(gcam_old.data - gcam).sum(dim=1).mean()

    return loss_att

def lf_dist(feat,feat_old, m_res=False): #,num_segments):
    if m_res:
        loss_1 = cl_utils.scale_loss(feat,feat_old,1)
        loss_2 = cl_utils.scale_loss(feat,feat_old,2)
        loss_4 = cl_utils.scale_loss(feat,feat_old,4)
        loss_8 = cl_utils.scale_loss(feat,feat_old,8)
        loss_dist = loss_1 + loss_2 + loss_4 + loss_8

    else:
        feat = feat.mean(1)
        feat_old = feat_old.mean(1)

        loss_dist = nn.CosineEmbeddingLoss()(feat,feat_old.clone().detach(),torch.ones(feat.shape[0]).cuda()).cuda()

    return loss_dist

def mr_loss(preds,targets,old_size,args):
    nb_negatives = args.mr_k
    if nb_negatives == 0:
        return torch.tensor(0).float().cuda()
    margin = args.margin
    gt_index = torch.zeros(preds.size()).cuda()
    gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
    gt_scores = preds.masked_select(gt_index)
    max_novel_scores = preds[:, old_size:].topk(nb_negatives, dim=1)[0]

    # the index of hard samples, i.e., samples of old classes
    hard_index = targets.lt(old_size)
    hard_num = torch.nonzero(hard_index).size(0)

    if hard_num > 0:
        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, nb_negatives)
        max_novel_scores = max_novel_scores[hard_index]
        assert (gt_scores.size() == max_novel_scores.size())
        assert (gt_scores.size(0) == hard_num)
        loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1), \
            max_novel_scores.view(-1, 1), torch.ones(hard_num*nb_negatives).cuda())
        return loss
    else:
        return torch.tensor(0).float().cuda()

def feat_dist(fmap, fmap_old, args, factor=None):
    factorize = args.factorize
    num_layers = len(fmap)
    loss_dist = torch.tensor(0.).cuda(non_blocking=True)
    for i in range(num_layers):
        f1 = fmap[i]
        f1 = f1.view(-1,args.num_segments,f1.size()[1],f1.size()[2],f1.size()[3]) # (B,T,C,H,W)
        f1 = f1.permute(0,2,1,3,4) # (B,C,T,H,W)
        f2 = fmap_old[i]
        f2 = f2.view(-1,args.num_segments,f2.size()[1],f2.size()[2],f2.size()[3])
        f2 = f2.permute(0,2,1,3,4)
        f1 = f1.pow(2)
        f2 = f2.pow(2)
        assert (f1.shape == f2.shape)
        B,C,T,H,W = f1.shape

        if factorize=='T-S':
            f_cur_t = cl_utils.factorize(f1,dim=2) # (B,C*H*W)
            f_old_t = cl_utils.factorize(f2,dim=2)
            f_cur_s = cl_utils.factorize(f1,dim=(3,4)) # (B,C*T)
            f_old_s = cl_utils.factorize(f2,dim=(3,4))
            f_cur = torch.cat([f_cur_t,f_cur_s],dim=-1)
            f_old = torch.cat([f_old_t,f_old_s],dim=-1)
            f_cur = F.normalize(f_cur, dim=1, p=2)
            f_old = F.normalize(f_old, dim=1, p=2)
            loss_i = torch.mean(torch.frobenius_norm(f_cur-f_old.clone().detach(),dim=-1))

        elif factorize=='T-GAP':
            f1 = cl_utils.factorize(f1,dim=(3,4)) # (B,C*T)
            f2 = cl_utils.factorize(f2,dim=(3,4))
            f1 = F.normalize(f1, dim=1, p=2)
            f2 = F.normalize(f2, dim=1, p=2)
            if factor is not None: # Ours
                factor_i = factor[i].permute(1,0)
                factor_i = factor[i].reshape([1,-1])
                loss_i = torch.mean(factor_i * torch.abs(f1-f2.clone().detach()))
            else:
                loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
            loss_i = loss_i/sqrt(T)

        elif factorize=='T-POD':
            f1_H = f1.sum(3) # B, C, T, W
            f1_W = f1.sum(4) # B, C, T, H
            f2_H = f2.sum(3)
            f2_W = f2.sum(4)
            f1 = torch.cat([f1_H,f1_W],dim=-1) # B, C, T, H+W
            f2 = torch.cat([f2_H,f2_W],dim=-1)
            f1 = F.normalize(f1, dim=-1, p=2)
            f2 = F.normalize(f2, dim=-1, p=2)
            f1 = f1.view(-1,C*T,f1.size()[3])
            f2 = f2.view(-1,C*T,f2.size()[3])
            if factor is not None: # Ours
                factor_i = factor[i].permute(1,0)
                factor_i = factor[i].reshape([1,-1])
                loss_i = torch.mean(factor_i * torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
            else:
                loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
            loss_i = loss_i/sqrt(T)


        elif factorize=='TH-TW':
            f_cur_th = cl_utils.factorize(f1,dim=(2,3)) # (B,C*H)
            f_old_th = cl_utils.factorize(f2,dim=(2,3))
            f_cur_tw = cl_utils.factorize(f1,dim=(2,4)) # (B,C*W)
            f_old_tw = cl_utils.factorize(f2,dim=(2,4))
            f_cur = torch.cat([f_cur_th,f_cur_th],dim=-1)
            f_old = torch.cat([f_old_tw,f_old_tw],dim=-1)
            f_cur = F.normalize(f_cur, dim=1, p=2)
            f_old = F.normalize(f_old, dim=1, p=2)
            loss_i = torch.mean(torch.frobenius_norm(f_cur-f_old.clone().detach(),dim=-1))

        elif factorize=='T-H-W':
            f_cur_t = cl_utils.factorize(f1,dim=2) # (B,C*H*W)
            f_old_t = cl_utils.factorize(f2,dim=2)
            f_cur_h = cl_utils.factorize(f1,dim=3) # (B,C*T*W)
            f_old_h = cl_utils.factorize(f2,dim=3)
            f_cur_w = cl_utils.factorize(f1,dim=4) # (B,C*T*H)
            f_old_w = cl_utils.factorize(f2,dim=4)

            f_cur = torch.cat([f_cur_t,f_cur_h,f_cur_w],dim=-1)
            f_old = torch.cat([f_old_t,f_old_h,f_old_w],dim=-1)
            f_cur = F.normalize(f_cur, dim=1, p=2)
            f_old = F.normalize(f_old, dim=1, p=2)
            loss_i = torch.mean(torch.frobenius_norm(f_cur-f_old.clone().detach(),dim=-1))

        elif factorize=='all':
            f1 = f1.view(B,-1) # B, C*T*H*W
            f2 = f2.view(B,-1)
            f1 = F.normalize(f1, dim=1, p=2)
            f2 = F.normalize(f2, dim=1, p=2)
            loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))

        elif factorize=='spatial_pixel':
            f1 = f1.reshape((B,C*T,-1))
            f2 = f2.reshape((B,C*T,-1))
            f1 = F.normalize(f1, dim=2, p=2)
            f2 = F.normalize(f2, dim=2, p=2)
            if factor is not None: # Ours
                factor_i = factor[i].permute(1,0)
                factor_i = factor_i.reshape([1,-1])
                loss_i = torch.mean(factor_i * torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
            else:
                loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
            loss_i = loss_i/sqrt(T)

        elif factorize=='pixel':
            #print(f1.size())
            #print(B,C,T,H,W)
            f1 = f1.reshape((B,C,-1)) # (B,C,T*H*W)
            f2 = f2.reshape((B,C,-1)) # (B,C,T*H*W)
            f1 = F.normalize(f1,dim=2,p=2)
            f2 = F.normalize(f2,dim=2,p=2)
            if factor is not None: 
                factor_i = factor[i].reshape([1,-1]) # (1,C)
                loss_i = torch.mean(factor_i * torch.frobenius_norm(f1-f2.clone().detach(), dim=-1))
            else:
                loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))

        loss_dist = loss_dist + loss_i

    loss_dist = loss_dist/num_layers

    return loss_dist

def nca_loss(similarities, targets, exclude_pos_denominator=True,
            hinge_proxynca=False, class_weights=None):

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")


def feat_corr(fmap, fmap_old, args):
    num_layers = len(fmap)
    loss_dist = torch.tensor(0.).cuda(non_blocking=True)
    for i in range(num_layers):
        f1 = fmap[i]
        f1 = f1.view(-1,args.num_segments,f1.size()[1],f1.size()[2],f1.size()[3])
        f1 = f1.permute(0,2,1,3,4)
        f2 = fmap_old[i]
        f2 = f2.view(-1,args.num_segments,f2.size()[1],f2.size()[2],f2.size()[3])
        f2 = f2.permute(0,2,1,3,4)

        f1 = f1.pow(2)
        f2 = f2.pow(2)
        assert (f1.shape == f2.shape)
        B,C,T,H,W = f1.shape

        f1_c = cl_utils.get_corr(f1)
        f2_c = cl_utils.get_corr(f2)
        f1_c = f1_c.view(B,-1) # B, C*T*H*W
        f2_c = f2_c.view(B,-1)
        f1_c = f1_c/(f1_c.norm(dim=1, keepdim=True) + 1e-8)
        f2_c = f2_c/(f2_c.norm(dim=1, keepdim=True) + 1e-8)
        loss_ci = f2_c.detach() - f1_c
        loss_ci = loss_ci.norm(dim=1).mean()
        loss_dist = loss_dist + loss_ci
    loss_dist = loss_dist/num_layers
    return loss_dist

def lf_dist_tcd(feat, feat_old, factor=None):
    B,T,C = feat.shape
    feat = F.normalize(feat, dim=-1,p=2)
    feat_old = F.normalize(feat_old, dim=-1,p=2)
    feat = feat.view(-1,T*C)
    feat_old = feat_old.view(-1,T*C)

    if factor is not None:
        #print(factor.size())
        factor = factor.reshape([1,-1])
        loss_dist = torch.mean(torch.sum(factor*((feat-feat_old)**2),1))
    else:
        loss_dist = torch.mean(torch.sum(((feat-feat_old)**2),1))
    loss_dist = loss_dist/T
    return loss_dist


def temporal_diversity_loss(feat, int_feat, num_segments, t_div_ratio=0.5):
    num_layers = len(int_feat)
    num_pairs = num_segments * (num_segments-1)
    loss_div = torch.tensor(0.).cuda(non_blocking=True)
    for i in range(num_layers):
        f = int_feat[i]
        _, C, H, W = f.shape
        if t_div_ratio < 1.0:
            C = int(C * t_div_ratio)
            f = f[:,:C,:,:]
        f = f.view(-1, num_segments, C, H, W) # (B, T, C', H,W)
        f = f.view(-1, num_segments, C, H*W)  # (B, T, C', H*W)
        f_1 = f.permute(0,2,1,3) # (B, C', T, H*W)
        f_1_norm = f_1.norm(p=2, dim=3, keepdim=True)
        f_2 = f.permute(0,2,3,1) # (B, C', H*W, T)
        f_2_norm = f_2.norm(p=2, dim=2, keepdim=True)

        R = torch.matmul(f_1,f_2)/(torch.matmul(f_1_norm, f_2_norm) + 1e-6) # (B, C', T, T)
        I = torch.eye(num_segments).cuda()
        I = I.repeat((R.shape[0], R.shape[1], 1, 1))

        #loss_i = F.relu(R-I).sum(-1).sum(-1) / num_pairs
        loss_i = F.frobenius_norm(R-I) / num_pairs
        loss_i = loss_i.mean()
        loss_div = loss_div + loss_i

    loss_div = loss_div/num_layers
    return loss_div




