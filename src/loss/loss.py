import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parallel import DistributedDataParallel as DDP

def weights_init(init_type='orthogonal'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class Contrastive(nn.Module):
    def __init__(self, args, netD, no_mlp=False, input_dim=1024, mlp_dim=256, temp=0.07):
        super(Contrastive, self).__init__()
        self.netD = netD
        self.l1 = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.no_mlp = no_mlp
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        if not no_mlp:
            self.mlp = nn.Sequential(*[nn.Linear(input_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, mlp_dim)])
            self.mlp.apply(weights_init)
            self.mlp.cuda()
            if args.distributed:
                self.mlp = DDP(self.mlp, device_ids= [args.local_rank], output_device=[args.local_rank])
        self.temp = temp

    def reduce_normalize(self, feat):
        if not self.no_mlp:
            feat = self.mlp(feat)               # reduce feat dim to [batch_size,256]
        norm = feat.pow(2).sum(1, keepdim=True).pow(1./2)
        out = feat.div(norm + 1e-7)
        return out

    def __call__(self, input, predict_result, gt, mask, num_feat_layers):
        # use more negative samples per positive sample
        gts = torch.repeat_interleave(gt, mask.size(0), dim=0)
        # print('mask\'s size: ', mask.size())
        masks = mask.repeat(mask.size(0), 1, 1, 1)
        inputs = (gts * (1 - masks).float()) + masks

        # handle features
        feat_outputs, feat_output_final, _, _ = self.netD(predict_result)
        feat_gts, feat_gt_final, _, _ = self.netD(gt)
        feat_corrupteds, _, _, _ = self.netD(input)
        _, feat_corrupted_final, _, _ = self.netD(inputs)

        # textural contrastive loss
        textural_loss = 0.
        assert num_feat_layers <= len(feat_gt_final)
        for idx in range(1, num_feat_layers):
            feat_output, feat_gt, feat_corrupted = feat_outputs[idx], feat_gts[idx], feat_corrupteds[idx]
            archor = feat_output
            pos = feat_gt
            neg = feat_corrupted
            textural_loss += self.l1(archor, pos) / (self.l1(archor, neg) + 1e-7)

        # semantic contrastive loss
        semantic_loss = 0.

        feat_output_final = self.reduce_normalize(feat_output_final)
        feat_gt_final = self.reduce_normalize(feat_gt_final)
        feat_corrupted_final = self.reduce_normalize(feat_corrupted_final)
        # print('feat_corrupted_final\'s size: ', feat_corrupted_final.size())
    
        batch_size = input.size(0)
        target_label = torch.zeros(batch_size, dtype=torch.long).cuda()
        anchor_final = feat_output_final.unsqueeze(1)
        # print('anchor_final\'s size: ', anchor_final.size())
        nce_matrix = []
        for i in range(batch_size):
            matrix = torch.cat([feat_gt_final[i].unsqueeze(0), 
            feat_corrupted_final[i*batch_size:(i+1)*batch_size]])
            nce_matrix.append(matrix.unsqueeze(0))
        nce_matrix = torch.cat(nce_matrix)
        # print('nce_matrix\'s size: ', nce_matrix.size())

        result = torch.bmm(anchor_final, nce_matrix.transpose(2, 1))
        result = result.view(batch_size, -1)
        # print('result\'s size: ', result.size())

        semantic_loss += self.cross_entropy_loss(result, target_label).mean()

        return textural_loss, semantic_loss


class L1(): 
    def __init__(self,):
        self.calc = nn.L1Loss()
    
    def __call__(self, x, y):
        return self.calc(x, y)

class dualadv():
    def __init__(self, args):
        self.globalgan_type = args.globalgan_type
        self.SCAT_type = args.SCAT_type

        if self.globalgan_type == 'nsgan':
            self.global_lossfn = nn.BCEWithLogitsLoss(reduction='none')
        elif self.globalgan_type == 'lsgan':
            self.global_lossfn = nn.MSELoss(reduction='none')

        if self.SCAT_type == 'nsgan':
            self.SCAT_lossfn = nn.BCEWithLogitsLoss(reduction='none')
        elif self.SCAT_type == 'lsgan':
            self.SCAT_lossfn = nn.MSELoss(reduction='none')
        
        if self.globalgan_type == 'hingegan' or self.SCAT_type == 'hingegan':
            self.relu = nn.ReLU()

    def D(self, netD, fake, real, masks):
        # 0s and 1s in masks represent missing and valid regions, respectively
        # standard global adversarial training
        fake_detach = fake.detach()
        _, _, d_fake_global, d_fake_pixel = netD(fake_detach)
        _, _, d_real_global, d_real_pixel = netD(real)

        if self.globalgan_type == 'nsgan' or self.globalgan_type == 'lsgan':
            real_label = torch.ones_like(d_real_global)
            fake_label = torch.zeros_like(d_fake_global)
            d_loss_global_real = self.global_lossfn(d_real_global, real_label).mean()
            d_loss_global_fake = self.global_lossfn(d_fake_global, fake_label).mean()
        
        elif self.globalgan_type == 'hingegan':
            d_loss_global_real = self.relu(1.0 - d_real_global).mean()
            d_loss_global_fake = self.relu(1.0 + d_fake_global).mean()
        
        d_loss_global = d_loss_global_real + d_loss_global_fake

        # segmentation confusion adversarial training
        if self.SCAT_type == 'nsgan' or self.SCAT_type == 'lsgan':
            real_label = torch.ones_like(d_real_pixel)
            fake_label = masks
            d_loss_SCAT_real = self.SCAT_lossfn(d_real_pixel, real_label).mean()
            d_loss_SCAT_fake = self.SCAT_lossfn(d_fake_pixel, fake_label).mean()
        
        elif self.SCAT_type == 'hingegan':
            d_loss_SCAT_real = self.relu(1.0 - d_real_pixel).mean()
            d_loss_SCAT_fake_valid = (self.relu(1.0 - d_fake_pixel)*masks).mean()      # valid regions in fake
            d_loss_SCAT_fake_generated = (self.relu(1.0 + d_fake_pixel)*(1-masks)).mean()   # generated regions in fake
            
            d_loss_SCAT_fake = d_loss_SCAT_fake_valid + d_loss_SCAT_fake_generated
        
        d_loss_SCAT = d_loss_SCAT_real + d_loss_SCAT_fake 

        return d_loss_global, d_loss_SCAT
    
    def G(self, netD, fake, masks):
        # standard global adversarial training
        _, _, g_fake_global, g_fake_pixel = netD(fake)
        if self.globalgan_type == 'nsgan' or self.globalgan_type == 'lsgan':
            real_label = torch.ones_like(g_fake_global)
            g_loss_global = self.global_lossfn(g_fake_global, real_label).mean()
        elif self.globalgan_type == 'hingegan':
            g_loss_global = -g_fake_global.mean()
        
        # segmentation confusion adversarial training
        if self.SCAT_type == 'nsgan' or self.SCAT_type == 'lsgan':
            real_label = torch.ones_like(g_fake_pixel)
            g_loss_SCAT = (self.SCAT_lossfn(g_fake_pixel, real_label)*(1-masks)).mean()
        elif self.SCAT_type == 'hingegan':
            g_loss_SCAT = (-g_fake_pixel*(1-masks)).mean()        

        return g_loss_global, g_loss_SCAT