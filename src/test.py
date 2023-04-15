from locale import normalize
import os
import argparse
import importlib
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from torchvision.utils import save_image
from pytorch_msssim import ssim
import math
import random
from itertools import islice
from utils.option import args 
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, size, shuffle=False):
        super(Dataset, self).__init__()
        img_tf = transforms.Compose(
        [#transforms.CenterCrop(size=(178, 178)), # use this for CelebA only 
        transforms.Resize(size=(size, size), 
        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        mask_tf = transforms.Compose(
            [transforms.Resize(size=size, interpolation=transforms.InterpolationMode.NEAREST), 
             transforms.ToTensor()])

        self.img_transform = img_tf
        self.mask_transform = mask_tf
        self.shuffle = shuffle

        self.paths = sorted(glob('{:s}/*'.format(img_root)))

        self.mask_paths = sorted(glob('{:s}/*.png'.format(mask_root)))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        if self.shuffle:
            mask_idx = random.randint(0, self.N_mask - 1)
        else:
            mask_idx = index if index < self.N_mask else index % (self.N_mask - 1)
        mask = Image.open(self.mask_paths[mask_idx])
        mask = self.mask_transform(mask.convert('L'))

        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        return gt_img, mask

    def __len__(self):
        return len(self.paths)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def L1(x, y):
    return nn.L1Loss()(x, y).item()

def normalize(x):
    x = x.transpose(1, 3) # [-1, 1]
    mean = torch.Tensor([1/2, 1/2, 1/2]).to(x.device)
    std = torch.Tensor([1/2, 1/2, 1/2]).to(x.device)
    x = x * std + mean # [0, 1]
    std = torch.Tensor([255, 255, 255]).to(x.device)   
    x = x * std # [0, 255]
    x = x.transpose(1, 3)
    return x

def main_worker(args): 
    
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()

    # prepare dataset
    dataset = Dataset(args.dir_image, args.dir_mask, args.image_size)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=args.shuffle, num_workers=4)
    image_data_loader = sample_data(dataloader)

    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'comp_results'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'gts'), exist_ok=True)
    
    # iteration through datasets
    num_imgs = 0
    L1error = 0.
    PSNR = 0.
    SSIM = 0. 
    num_imgs = 0

    for idx in tqdm(range(args.num_test)):
        image, mask = next(image_data_loader)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask).float() + mask 
        with torch.no_grad():
            pred_img = model(image_masked, mask)
        
        comp_imgs = (1 - mask) * image + mask * pred_img
        L1error += L1(comp_imgs, image)

        save_image(torch.cat([image_masked, comp_imgs, image], 0), os.path.join(args.outputs, f'{idx}_all.jpg'), nrow=3, normalize=True, value_range=(-1, 1))
        save_image(mask, os.path.join(args.outputs, f'{idx}_hole.jpg'), normalize=True, value_range=(0, 1))        
        save_image(image_masked, os.path.join(args.outputs, f'{idx}_masked.jpg'), normalize=True, value_range=(-1, 1))   
        save_image(comp_imgs, os.path.join(args.outputs, f'{idx}_comp.jpg'), normalize=True, value_range=(-1, 1))     
        save_image(comp_imgs, os.path.join(args.outputs, 'comp_results', f'{idx}.png'), normalize=True, value_range=(-1, 1))
        save_image(image, os.path.join(args.outputs, f'{idx}_gt.jpg'), normalize=True, value_range=(-1, 1))  
        save_image(image, os.path.join(args.outputs, 'gts', f'{idx}.png'), normalize=True, value_range=(-1, 1))                              
       
        num_imgs += 1

        comp_imgs = normalize(comp_imgs)
        image = normalize(image)
        ssim_result = ssim(comp_imgs, image, data_range=255, size_average=True).item()
        SSIM += ssim_result
        mse = torch.pow(comp_imgs - image, 2).mean().item()
        PSNR += 10*math.log10(255**2/mse)
    
    print('PSNR:%.2f' % (PSNR/args.num_test))
    print('SSIM:%.4f' % (SSIM/args.num_test))
    print('L1error:%.6f' % (L1error/args.num_test))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main_worker(args)
