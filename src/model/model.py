import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn.utils import spectral_norm
from .common import BaseNetwork

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class SNUpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(SNUpConv, self).__init__()
        self.scale = scale
        self.snconv = spectral_norm(nn.Conv2d(inc, outc, 3, stride=1, padding=1))

    def forward(self, x):
        return self.snconv(F.interpolate(x, scale_factor=2, mode='nearest'))

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class AOTBlock(nn.Module):
    '''
    A resblock equipped with different dilated convolutions for better context encoding.
    https://github.com/researchmm/AOT-GAN-for-Inpainting/blob/master/src/model/aotgan.py#L54
    '''    
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask

class InpaintGenerator(BaseNetwork):
    '''
    AOT-GAN generator backbone.
    https://github.com/researchmm/AOT-GAN-for-Inpainting/blob/master/src/model/aotgan.py#L9
    '''      
    def __init__(self, args):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


# ----- U-net discriminator -----
class UnetDiscriminator(BaseNetwork):
    def __init__(self, downnl='lrelu', upnl='relu'):
        super(UnetDiscriminator, self).__init__()
        down_nl_layer = get_non_linearity(layer_type=downnl)
        up_nl_layer = get_non_linearity(layer_type=upnl)
        inc = 3
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1)),
            down_nl_layer()) # output size 128^2x64
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            down_nl_layer()) # output size 64^2x128   
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            down_nl_layer()) # output size 32^2x256     
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1)),
            down_nl_layer()) # output size 16^2x512  
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 1024, 4, stride=2, padding=1)),
            down_nl_layer()) # output size 8^2x1024 

        self.conv_bottleneck = nn.Sequential(
            spectral_norm(nn.Conv2d(1024, 1024, 4, stride=2, padding=1)),
            down_nl_layer()) # output size 4^2x1024 

        self.globalavgpool = nn.AvgPool2d(4) # output size 1^2x1024

        self.fc = spectral_norm(nn.Linear(1024, 1))

        self.upconv_bottleneck = nn.Sequential(
            SNUpConv(1024, 1024),
            up_nl_layer()) # output size 8^2x1024

        self.upconv5 = nn.Sequential(
            SNUpConv(1024*2, 512),
            up_nl_layer()) # output size 16^2x512
        self.upconv4 = nn.Sequential(
            SNUpConv(512*2, 256),
            up_nl_layer()) # output size 32^2x256
        self.upconv3 = nn.Sequential(
            SNUpConv(256*2, 128),
            up_nl_layer()) # output size 64^2x128        
        self.upconv2 = nn.Sequential(
            SNUpConv(128*2, 64),
            up_nl_layer()) # output size 128^2x64
        self.upconv1 = nn.Sequential(
            SNUpConv(64*2, 64),
            up_nl_layer()) # output size 256^2x64
        self.finalconv = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.init_weights()

    def forward(self, x):
        # encoder stage
        feat1 = self.conv1(x)      # output size 128^2x64
        feat2 = self.conv2(feat1)  # output size 64^2x128
        feat3 = self.conv3(feat2)  # output size 32^2x256
        feat4 = self.conv4(feat3)  # output size 16^2x512
        feat5 = self.conv5(feat4)  # output size 8^2x1024
        bottleneck = self.conv_bottleneck(feat5) # output size 4^2x1024
        bottleneck_avgpool = self.globalavgpool(bottleneck) # output size 1x1x1024
        last_feat = bottleneck_avgpool.view(-1, bottleneck_avgpool.size(1)) # output size 1024
        one_dim = self.fc(last_feat) # output size 1

        # decoder stage
        up_bottleneck = self.upconv_bottleneck(bottleneck)            # output size 8^2x1024
        up_feat5 = self.upconv5(torch.cat([up_bottleneck, feat5], 1)) # output size 16^2x512
        up_feat4 = self.upconv4(torch.cat([up_feat5, feat4], 1))      # output size 32^2x256
        up_feat3 = self.upconv3(torch.cat([up_feat4, feat3], 1))      # output size 64^2x128
        up_feat2 = self.upconv2(torch.cat([up_feat3, feat2], 1))      # output size 128^2x64
        up_feat1 = self.upconv1(torch.cat([up_feat2, feat1], 1))      # output size 256^2x64
        pixel_output = self.finalconv(up_feat1)
        return last_feat, one_dim, pixel_output

# ----- ResU-net discriminator -----
class ResUnetDiscriminator(BaseNetwork):
    def __init__(self, ):
        super(ResUnetDiscriminator, self).__init__()
        inc = 3
        self.init_weights()

    def forward(self, x):
        pass