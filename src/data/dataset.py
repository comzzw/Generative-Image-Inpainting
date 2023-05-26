import os
import numpy as np
from glob import glob
from PIL import Image

import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from data.mask_generator import RandomMask

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)

class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type
        
        # image and mask 
        self.image_path = []
        if args.scan_subdirs:
            self.paths = self.make_dataset_from_subdirs(args.dir_image)
        else:
            self.paths = [entry.path for entry in os.scandir(args.dir_image) 
                                              if is_image_file(entry.name)]
        self.image_path.extend(self.paths)                                              
        if self.mask_type == 'pconv':
            self.mask_path = glob(os.path.join(args.dir_mask, '*.png'))

        # augmentation
        if args.transform == 'randomcrop':
            self.img_trans = transforms.Compose(
                [transforms.RandomResizedCrop(args.crop_size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))
                 ]
            )
        elif args.transform == 'centercrop':
            self.img_trans = transforms.Compose(
                [transforms.CenterCrop(args.crop_size),
                 transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))
                 ]
            )            
        elif args.transform == 'resize_and_crop':
            self.img_trans = transforms.Compose(
                [transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                 transforms.CenterCrop(args.crop_size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))
                 ]
            )            
        else:
            raise NotImplementedError("Image transformation type %s is not implemented!" % args.transform)

        self.mask_trans = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        self.mask_trans_simple = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, dirs, _ in os.walk(folder_path, followlinks=True):
            for dir in dirs:
                dir = os.path.join(root, dir)
                sub_samples = make_dataset_from_current_dir(dir)
                samples += sub_samples

        def make_dataset_from_current_dir(folder_path):
            samples = []
            for root, _, fnames in os.walk(folder_path, followlinks=True):
                for fname in fnames:
                    if is_image_file(fname):
                        samples.append(os.path.join(root, fname))
            return samples
    
        return samples

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        image = self.img_trans(image)
        filename = os.path.basename(self.image_path[index])

        if self.mask_type == 'pconv':
            index = np.random.randint(0, len(self.mask_path))
            mask = Image.open(self.mask_path[index])
            mask = mask.convert('L')
            mask = self.mask_trans(mask)
        elif self.mask_type == 'centered':
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[self.h//4:self.h//4*3, self.w//4:self.w//4*3] = 1
            mask = Image.fromarray(mask).convert('L')
            mask = self.mask_trans(mask)
        elif self.mask_type == 'random':
            mask = Image.fromarray(RandomMask(self.h)).convert('L')
            mask = self.mask_trans_simple(mask)

        return image, mask, filename