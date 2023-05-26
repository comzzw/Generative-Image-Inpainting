import functools
from model.common import BaseNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=False)
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

# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, use_SN=True, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.to_q = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.to_key = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.to_value = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.output = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_SN:
            self.to_q, self.to_key, self.to_value, self.output = spectral_norm(self.to_q), spectral_norm(self.to_key), spectral_norm(self.to_value), spectral_norm(self.output)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
    def forward(self, x, y=None):
        # Apply convs
        query = self.to_q(x)
        # downsample key and value's spatial size
        key = F.max_pool2d(self.to_key(x), [2,2])
        value = F.max_pool2d(self.to_value(x), [2,2])
        # Perform reshapes
        query = query.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        key = key.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        value = value.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        attn_map = F.softmax(torch.bmm(query.transpose(1, 2), key), -1)
        # Attention map times g path
        o = self.output(torch.bmm(value, attn_map.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

# Conv block for the discriminator encoder
class EBlock(nn.Module):
    def __init__(self, inc, outc, use_SN=True, activation='relu'):
        super(EBlock, self).__init__()
        self.inc, self.outc = inc, outc

        self.use_SN = use_SN
        self.activation = get_non_linearity(activation)()

        # Conv layer
        self.conv = nn.Conv2d(self.inc, self.outc, 4, stride=2, padding=1)

        if self.use_SN:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.activation(self.conv(x))

# Conv block for the discriminator decoder
class DBlock(nn.Module):
    def __init__(self, inc, outc, use_SN=True, activation='relu'):
        super(DBlock, self).__init__()
        self.inc, self.outc = inc, outc

        self.use_SN = use_SN
        self.activation = get_non_linearity(activation)()

        # Conv layer
        self.conv = nn.Conv2d(self.inc, self.outc, 3, stride=1, padding=1)

        if self.use_SN:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.activation(self.conv(F.interpolate(x, scale_factor=2, mode='nearest')))

# Residual block for the discriminator encoder
class EResBlock(nn.Module):
    def __init__(self, inc, outc, use_SN=True, wide=True,
                preactivation=False, activation='relu', downsample=None):
        super(EResBlock, self).__init__()
        self.inc, self.outc = inc, outc
        self.hc = self.outc if wide else self.inc
        self.use_SN = use_SN
        self.preactivation = preactivation
        self.activation = get_non_linearity(activation)()
        self.downsample = downsample

        # Conv layers
        self.conv1 = nn.Conv2d(self.inc, self.hc, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.hc, self.outc, 3, stride=1, padding=1)
        self.learnable_sc = True if (inc != outc) or downsample else False
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(self.inc, self.outc, 1, stride=1, padding=0)
        
        if self.use_SN:
            self.conv1, self.conv2, self.conv_sc = spectral_norm(self.conv1), spectral_norm(self.conv2), spectral_norm(self.conv_sc)
    # skip connection
    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x
        
    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it 
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x    
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)     

        return h + self.shortcut(x)

# Residual block for the discriminator decoder
class DResBlock(nn.Module):
    def __init__(self, inc, outc, use_SN=True, activation='relu',
                upsample=None, skip_connection=True):
        super(DResBlock, self).__init__()

        self.inc, self.outc = inc, outc
        self.use_SN = use_SN
        self.activation = get_non_linearity(activation)()
        self.upsample = upsample

        # Conv layers
        self.conv1 = nn.Conv2d(self.inc, self.outc, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.outc, self.outc, 3, stride=1, padding=1)    
        self.learnable_sc = inc != outc or upsample
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(inc, outc, 1, stride=1, padding=0)

        if self.use_SN:
            self.conv1, self.conv2, self.conv_sc = spectral_norm(self.conv1), spectral_norm(self.conv2), spectral_norm(self.conv_sc)

        # upsample layers
        self.upsample = upsample
        self.skip_connection = skip_connection

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        #print(h.size())
        h = self.activation(h)
        h = self.conv2(h)
        # may be changed to h = self.conv2.forward_wo_sn(h)
        if self.learnable_sc:
            x = self.conv_sc(x)


        if self.skip_connection:
            out = h + x
        else:
            out = h
        return out

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
        self.print_network()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x
    

def D_unet_arch(ch=64, attention='64'):
    arch = {}

    arch[256] = {'in_channels' :            [3] + [ch*item for item in [1, 2, 4, 8, 8, 16, 8*2, 8*2, 4*2, 2*2, 1*2  , 1         ]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 8,   8,   4,   2,   1,   1          ]],
                             'downsample' : [True] *6 + [False]*6 ,
                             'upsample':    [False]*6 + [True] *6,
                             'resolution' : [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,13)}}



    return arch

# ----- A simple U-net discriminator -----
class UnetDiscriminator(BaseNetwork):

    def __init__(self, args, D_ch=64, 
                             D_attn='64', D_activation='relu',
                             output_dim=1, D_init='orthogonal', **kwargs):
        super(UnetDiscriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Resolution
        self.resolution = args.image_size
        # Attention?
        self.use_attn = args.use_D_attn
        self.attention = D_attn
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.use_SN = not args.no_SN

        self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4        
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention)[self.resolution]
        
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[EBlock(inc=self.arch['in_channels'][index],
                                                         outc=self.arch['out_channels'][index],
                                                         use_SN=self.use_SN,
                                                         activation=self.activation)]]

            elif self.arch["upsample"][index]:

                self.blocks += [[DBlock(inc=self.arch['in_channels'][index],
                                                         outc=self.arch['out_channels'][index],
                                                         use_SN=self.use_SN,
                                                         activation=self.activation)]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition and self.use_attn: #index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.use_SN)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])


        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier,1,kernel_size=1)
        self.blocks.append(last_layer)

        self.linear_middle = nn.Linear(16*self.ch, output_dim)
        if self.use_SN:
            self.linear_middle = spectral_norm(self.linear_middle)


        self.init_weights(init_type=self.init)
        self.print_network()

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):

            if index==7:
                h = torch.cat((h,residual_features[5]),dim=1)
            elif index==8:
                h = torch.cat((h,residual_features[4]),dim=1)
            elif index==9:#
                h = torch.cat((h,residual_features[3]),dim=1)
            elif index==10:#
                h = torch.cat((h,residual_features[2]),dim=1)
            elif index==11:
                h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(get_non_linearity(self.activation)()(h), [2, 3])
                bottleneck_out = self.linear_middle(h_)

        out = self.blocks[-1](h)

        out = out.view(out.size(0),1,self.resolution,self.resolution)

        return residual_features, h_, bottleneck_out, out

class ResUnetDiscriminator(BaseNetwork):

    def __init__(self, args, D_ch=64, D_wide=True, 
                             D_attn='64', D_activation='relu',
                             output_dim=1, D_init='orthogonal', **kwargs):
        super(ResUnetDiscriminator, self).__init__()


        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = args.image_size
        # Attention?
        self.use_attn = args.use_D_attn
        self.attention = D_attn
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.use_SN = not args.no_SN



        self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention)[self.resolution]
       # print(self.arch)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[EResBlock(inc=self.arch['in_channels'][index],
                                             outc=self.arch['out_channels'][index],
                                             use_SN=self.use_SN,
                                             wide=self.D_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                    if self.arch['upsample'][index] else None)

                self.blocks += [[DResBlock(inc=self.arch['in_channels'][index],
                                                         outc=self.arch['out_channels'][index],
                                                         use_SN=self.use_SN,
                                                         activation=self.activation,
                                                         upsample= upsample_function, skip_connection = True )]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition and self.use_attn: #index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.use_SN)]


        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])


        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier,1,kernel_size=1)
        self.blocks.append(last_layer)

        self.linear_middle = nn.Linear(16*self.ch, output_dim)
        if self.use_SN:
            self.linear_middle = spectral_norm(self.linear_middle)


        self.init_weights(init_type=self.init)

        self.print_network()

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):

            if index==7:
                h = torch.cat((h,residual_features[5]),dim=1)
            elif index==8:
                h = torch.cat((h,residual_features[4]),dim=1)
            elif index==9:#
                h = torch.cat((h,residual_features[3]),dim=1)
            elif index==10:#
                h = torch.cat((h,residual_features[2]),dim=1)
            elif index==11:
                h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(get_non_linearity(self.activation)()(h), [2, 3])
                bottleneck_out = self.linear_middle(h_)

        out = self.blocks[-1](h)

        out = out.view(out.size(0),1,self.resolution,self.resolution)

        return residual_features, h_, bottleneck_out, out

if __name__ == '__main__':
    class args():
        def __init__(self) -> None:
            pass
        image_size = 256
        rates = [1,2,4,8]
        block_num = 8
        use_D_attn = False
        no_SN = False

    simpleUNet = UnetDiscriminator(args)
    ResUNet = ResUnetDiscriminator(args)    
    generator = InpaintGenerator(args)
    img = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    pred_img = generator(img, mask)
    residual_features, h_, bottleneck_out, out = simpleUNet(img)
