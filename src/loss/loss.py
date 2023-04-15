import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, netD, no_mlp=False, input_dim=1024, mlp_dim=256, temp=0.07):
        super(Contrastive, self).__init__()
        self.discriminator = netD
        self.l1 = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.no_mlp = no_mlp
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        if not no_mlp:
            self.mlp = nn.Sequential(*[nn.Linear(input_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, mlp_dim)])
            self.mlp.apply(weights_init)
            self.mlp.cuda()
        self.temp = temp

    def reduce_normalize(self, feat):
        if not self.no_mlp:
            feat = self.mlp(feat)               # reduce feat dim to [batch_size,256]
        norm = feat.pow(2).sum(1, keepdim=True).pow(1./2)
        out = feat.div(norm + 1e-7)
        return out

    def __call__(self, input, predict_result, gt, mask, num_feat_layers, start_layer_id=1):
        # textural contrastive loss
        feat_outputs, feat_gts, feat_corrupteds = [], [], []
        textural_loss = 0.
        for idx in range(num_feat_layers):
            if idx == 0:
                feat_outputs.append(predict_result)
                feat_gts.append(gt)
                feat_corrupteds.append(input)
            else:
                enc_layer = getattr(self.discriminator, 'conv%d' % idx)
                feat_output, feat_gt, feat_corrupted = enc_layer(feat_outputs[-1]), enc_layer(feat_gts[-1]), enc_layer(feat_corrupteds[-1])
                feat_outputs.append(feat_output)
                feat_gts.append(feat_gt)
                feat_corrupteds.append(feat_corrupted)
            if idx >= start_layer_id:
                batch_size = input.size(0)
                archor = feat_outputs[idx]
                pos = feat_gts[idx]
                neg = feat_corrupteds[idx]
                textural_loss += self.l1(archor, pos) / (self.l1(archor, neg) + 1e-7)

        # semantic contrastive loss
        semantic_loss = 0.
        # use more negative samples per positive sample
        gts = torch.repeat_interleave(gt, mask.size(0), dim=0)
        # print('mask\'s size: ', mask.size())
        masks = mask.repeat(mask.size(0), 1, 1, 1)
        inputs = (gts * (1 - masks).float()) + masks

        feat_output_final, _, _ = self.discriminator(predict_result)
        feat_gt_final, _, _ = self.discriminator(gt)
        feat_corrupteds_final, _, _ = self.discriminator(inputs)
        # print('feat_comp_final\'s size: ', feat_comp_final.size())

        feat_output_final = self.reduce_normalize(feat_output_final)
        feat_gt_final = self.reduce_normalize(feat_gt_final)
        feat_corrupteds_final = self.reduce_normalize(feat_corrupteds_final)
        # print('feat_comp_final\'s size: ', feat_comp_final.size())
    
        batch_size = input.size(0)
        target_label = torch.zeros(batch_size, dtype=torch.long).cuda()
        anchor_final = feat_output_final.unsqueeze(1)
        # print('anchor_final\'s size: ', anchor_final.size())
        nce_matrix = []
        for i in range(batch_size):
            matrix = torch.cat([feat_gt_final[i].unsqueeze(0), 
            feat_corrupteds_final[i*batch_size:(i+1)*batch_size]])
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
        self.calc = torch.nn.L1Loss()
    
    def __call__(self, x, y):
        return self.calc(x, y)

class hingegan():
    def __init__(self, ):
        self.relu = nn.ReLU()
    
    def D(self, netD, fake, real):
        fake_detach = fake.detach()
        _, d_fake, _ = netD(fake_detach)
        _, d_real, _ = netD(real)

        d_loss_real = self.relu(1.0 - d_real).mean()
        d_loss_fake = self.relu(1.0 + d_fake).mean()
        dis_loss = d_loss_fake + d_loss_real
        
        return dis_loss
    
    def G(self, netD, fake):
        _, g_fake, _ = netD(fake)
        gen_loss = -g_fake.mean()

        return gen_loss

class masked_hingegan():
    def __init__(self, ):
        self.relu = nn.ReLU()
    
    def D(self, netD, fake, real, masks):
        fake_detach = fake.detach()
        _, _, d_fake = netD(fake_detach)
        _, _, d_real = netD(real)
        
        d_loss_real = self.relu(1.0 - d_real).mean()
        d_loss_real_ = (self.relu(1.0 - d_fake)*masks).mean()      # valid regions in fake
        d_loss_fake = (self.relu(1.0 + d_fake)*(1-masks)).mean()   # generated regions in fake
        dis_loss = d_loss_fake + d_loss_real_ + d_loss_real
        
        return dis_loss
    
    def G(self, netD, fake, masks):
        _, _, g_fake = netD(fake)
        gen_loss = -g_fake*(1-masks)

        return gen_loss.mean()

class nsgan(): 
    def __init__(self, ):
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def D(self, netD, fake, real):
        fake_detach = fake.detach()
        _, d_fake, _ = netD(fake_detach)
        _, d_real, _ = netD(real)
        real_label = torch.ones_like(d_real)
        fake_label = torch.zeros_like(d_fake)
        dis_loss = self.loss_fn(d_real, real_label) + self.loss_fn(d_fake, fake_label)
        return dis_loss

    def G(self, netD, fake):
        _, g_fake, _ = netD(fake)
        real_label = torch.ones_like(g_fake)
        gen_loss = self.loss_fn(g_fake, real_label)
        return gen_loss

class masked_nsgan(): 
    def __init__(self, ):
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def D(self, netD, fake, real, masks):
        fake_detach = fake.detach()
        _, _, d_fake = netD(fake_detach)
        _, _, d_real = netD(real)
        real_label = torch.ones_like(d_real)
        mask_label = masks
        dis_loss = self.loss_fn(d_real, real_label) + self.loss_fn(d_fake, mask_label)
        return dis_loss

    def G(self, netD, fake, masks):
        _, _, g_fake = netD(fake)
        real_label = torch.ones_like(g_fake)
        gen_loss = self.loss_fn(g_fake, real_label)*(1-masks).mean()
        return gen_loss