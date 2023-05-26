import argparse

parser = argparse.ArgumentParser(description='Image Inpainting')

# data specifications 
parser.add_argument('--dir_image', type=str, default='../../dataset',
                    help='image dataset directory for training or testing')
parser.add_argument('--scan_subdirs', action='store_true',
                    help='scan subdirs for images, e.g., Places2 dataset')
parser.add_argument('--dir_mask', type=str, default='../../dataset',
                    help='mask dataset directory for training or testing')               
parser.add_argument('--ipath', type=str, default='../../dataset',
                    help='image path for single image test')
parser.add_argument('--mpath', type=str, default='../../dataset',
                    help='mask path for single image test')   
parser.add_argument('--dataset', type=str, default='Places2',
                    help='training dataset: (Places2 | CelebA)')
parser.add_argument('--image_size', type=int, default=256,
                    help='image size used during training')
parser.add_argument('--crop_size', type=int, default=256,
                    help='image crop size used during training, use 178 for CelebA dataset')
parser.add_argument('--transform', type=str, default='randomcrop',
                    help='image transformation type: (randomcrop | centercrop | resize_and_crop), use centercrop and randomcrop for CelebA and Places2 respectively')
parser.add_argument('--mask_type', type=str, default='pconv',
                    help='mask used during training: (pconv| centered | random), pconv needs to specify --dir_mask')


# model specifications 
parser.add_argument('--model', type=str, default='model',
                    help='model name')
parser.add_argument('--block_num', type=int, default=8,
                    help='number of AOT blocks')
parser.add_argument('--rates', type=str, default='1+2+4+8',
                    help='dilation rates used in AOT block')
parser.add_argument('--netD', type=str, default='Unet',
                    help='discriminator network: (Unet | ResUnet)')
parser.add_argument('--use_D_attn', action='store_true',
                    help='use self-attention in netD')    
parser.add_argument('--no_SN', action='store_true',
                    help='not use spectral normalization in netD')    
parser.add_argument('--globalgan_type', type=str, default='hingegan',
                    help='global adversarial training: (hingegan | nsgan | lsgan)')                                
parser.add_argument('--SCAT_type', type=str, default='hingegan',
                    help='segmentation confusion adversarial training: (hingegan | nsgan | lsgan)')
parser.add_argument('--no_mlp', action='store_true',
                    help='use mlp for semantic contrastive loss')               

# hardware specifications 
parser.add_argument('--seed', type=int, default=77,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers used in data loader')

# optimization specifications 
parser.add_argument('--lrg', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lrd', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0,
                    help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='beta2 in optimier')

# loss specifications 
parser.add_argument('--rec_loss', type=str, default='10*L1',
                    help='losses for reconstruction')                    
parser.add_argument('--adv_weight', type=float, default=1,
                    help='loss weight for adversarial losses')          
parser.add_argument('--text_weight', type=float, default=10,
                    help='loss weight for textural contrastive loss')  
parser.add_argument('--sem_weight', type=float, default=1,
                    help='loss weight for semantic contrastive loss')            

# training specifications 
parser.add_argument('--iterations', type=int, default=1e5,
                    help='the number of iterations for training')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size in each mini-batch')
parser.add_argument('--port', type=int, default=22334,
                    help='tcp port for distributed training')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous iteration')


# log specifications 
parser.add_argument('--print_every', type=int, default=10,
                    help='frequency for updating progress bar')
parser.add_argument('--save_every', type=int, default=1e4,
                    help='frequency for saving models')
parser.add_argument('--save_dir', type=str, default='../experiments',
                    help='directory for saving models and logs')
parser.add_argument('--tensorboard', action='store_true',
                    help='default: false, since it will slow training. use it for debugging')

# test specifications   
parser.add_argument('--pre_train', type=str, default=None,
                    help='path to pretrained models')
parser.add_argument('--outputs', type=str, default='../outputs', 
                    help='path to save results')
parser.add_argument('--num_test',  type=int, default=100, 
                    help='number of test images')        
parser.add_argument('--shuffle', action='store_true',
                    help='sample random images for testing')                   


# ----------------------------------
args = parser.parse_args()
args.iterations = int(args.iterations)

args.rates = list(map(int, list(args.rates.split('+'))))

losses = list(args.rec_loss.split('+'))
args.rec_loss = {}
for l in losses: 
    weight, name = l.split('*')
    args.rec_loss[name] = float(weight)