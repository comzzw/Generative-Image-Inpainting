import os
import importlib
from PIL import Image
from glob import glob
from itertools import islice
from utils.option import args 

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


def main_worker(args, use_gpu=True): 
    # load image and mask
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    img_tf = transforms.Compose(
    [# transforms.CenterCrop(size=(178, 178)), # use this for CelebA only 
    transforms.Resize(size=(args.image_size, args.image_size),
    interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    mask_tf = transforms.Compose(
        [transforms.Resize(size=args.image_size,
        interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])
    
    image = img_tf(Image.open(args.ipath).convert('RGB'))
    mask = mask_tf(Image.open(args.mpath).convert('L'))

    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
 
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()

    # prepare output folder
    os.makedirs(args.outputs, exist_ok=True)
    
    image, mask = image.cuda().unsqueeze(0), mask.cuda().unsqueeze(0)

    # print(mask.size(), image.size())
    image_masked = image * (1 - mask).float() + mask 
    with torch.no_grad():
        pred_img = model(image_masked, mask)

    comp_imgs = (1 - mask) * image + mask * pred_img
    image_name = os.path.basename(args.ipath).split('.')[0]
    mask_name = os.path.basename(args.mpath).split('.')[0]

    save_image(torch.cat([image_masked, comp_imgs, image], 0), 
    os.path.join(args.outputs, f'{image_name}_{mask_name}_all.png'), nrow=3, normalize=True)
    save_image(image_masked, os.path.join(args.outputs, f'{image_name}_{mask_name}masked.png'), normalize=True)
    save_image(comp_imgs, os.path.join(args.outputs, f'{image_name}_{mask_name}_comp.png'), normalize=True)     


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main_worker(args)
