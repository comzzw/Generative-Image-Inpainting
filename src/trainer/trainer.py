import os
import importlib
from tqdm import tqdm
from glob import glob

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from data import create_loader 
from loss import loss as loss_module
from trainer.common import timer, reduce_loss_dict


class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        # setup data set and data loader
        self.dataloader, self.dataloader_generator = create_loader(args)

        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        
        self.dualadv_loss = loss_module.dualadv(args)

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        net = importlib.import_module('model.'+args.model)
        
        self.netG = net.InpaintGenerator(args).cuda()
        if args.netD == 'Unet':
            self.netD = net.UnetDiscriminator(args).cuda()
        elif args.netD == 'ResUnet':
            self.netD = net.ResUnetDiscriminator(args).cuda()
        self.contrastive_loss = getattr(loss_module, 'Contrastive')(args, self.netD, args.no_mlp)
        
        if args.distributed:
            self.netG = DDP(self.netG, device_ids= [args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids= [args.local_rank], output_device=[args.local_rank])
            if not args.no_mlp:
                self.contrastive_loss.mlp = DDP(self.contrastive_loss.mlp, device_ids= [args.local_rank], output_device=[args.local_rank])

        if not args.no_mlp:
            self.optimG = torch.optim.Adam(
                list(self.netG.parameters())+list(self.contrastive_loss.mlp.parameters()), lr=args.lrg, betas=(args.beta1, args.beta2))    
        else:
            self.optimG = torch.optim.Adam(
                self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))                    
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))
        
        if args.resume:
            self.load()     

        if args.tensorboard: 
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            

    def load(self):
        try: 
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'ckpt', 'G*.pt'))))[-1]
            if isinstance(self.netG, DDP):
                self.netG.module.load_state_dict(torch.load(gpath, map_location='cuda'))
            else:
                self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0: 
                print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 

        try: 
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'ckpt', 'D*.pt'))))[-1]
            data = torch.load(dpath, map_location='cuda')
            if not self.args.no_mlp:
                if isinstance(self.netD, DDP):
                    self.netD.module.load_state_dict(data['netD'])
                    self.contrastive_loss.mlp.module.load_state_dict(data['MLP'])
                else:
                    self.netD.load_state_dict(data['netD'])
                    self.contrastive_loss.mlp.load_state_dict(data['MLP'])
                if self.args.global_rank == 0: 
                    print(f'[**] Loading discriminator and mlp network from {dpath}')
            else:
                if isinstance(self.netD, DDP):
                    self.netD.module.load_state_dict(data)
                else:
                    self.netD.load_state_dict(data)
                if self.args.global_rank == 0: 
                    print(f'[**] Loading discriminator network from {dpath}')                
        except: 
            pass
        
        try: 
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'ckpt', 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.global_rank == 0: 
                print(f'[**] Loading optimizer from {opath}')
        except: 
            pass


    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            if isinstance(self.netG, DDP):
                torch.save(self.netG.module.state_dict(),                       
                    os.path.join(self.args.save_dir, 'ckpt', f'G{str(self.iteration).zfill(7)}.pt'))           
            else:
                torch.save(self.netG.state_dict(), 
                    os.path.join(self.args.save_dir, 'ckpt', f'G{str(self.iteration).zfill(7)}.pt'))                              
            if isinstance(self.netD, DDP):
                if not self.args.no_mlp:
                    torch.save(
                        {'netD': self.netD.module.state_dict(), 'MLP': self.contrastive_loss.mlp.module.state_dict()}, 
                        os.path.join(self.args.save_dir, 'ckpt', f'D{str(self.iteration).zfill(7)}.pt'))
                else:
                    torch.save(self.netD.module.state_dict(), 
                        os.path.join(self.args.save_dir, 'ckpt', f'D{str(self.iteration).zfill(7)}.pt'))  
            else:
                if not self.args.no_mlp:
                    torch.save(
                        {'netD': self.netD.state_dict(), 'MLP': self.contrastive_loss.mlp.state_dict()}, 
                        os.path.join(self.args.save_dir, 'ckpt', f'D{str(self.iteration).zfill(7)}.pt'))
                else:
                    torch.save(self.netD.state_dict(), 
                        os.path.join(self.args.save_dir, 'ckpt', f'D{str(self.iteration).zfill(7)}.pt'))                  

            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()}, 
                os.path.join(self.args.save_dir, 'ckpt', f'O{str(self.iteration).zfill(7)}.pt'))
            

    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        if self.args.global_rank == 0: 
            pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
            timer_data, timer_model = timer(), timer()

        num_epoch_iter = len(self.dataloader) // (self.args.batch_size * self.args.world_size)
        current_epoch = self.iteration // num_epoch_iter

        for idx in pbar:
            self.iteration += 1
            if self.args.distributed and (self.iteration-1) % num_epoch_iter == 0:
                current_epoch += 1
                self.dataloader.sampler.set_epoch(current_epoch)

            images, masks, filename = next(self.dataloader_generator)
            images, masks = images.cuda(), masks.cuda()
            images_masked = (images * (1 - masks).float()) + masks

            if self.args.global_rank == 0: 
                timer_data.hold()
                timer_model.tic()

            # in: [rgb(3) + mask(1)]
            pred_img = self.netG(images_masked, masks)
            comp_img = (1 - masks) * images + masks * pred_img

            losses = {}
            D_losses = {}
            G_losses = {}

            # optimize D
            
            D_losses[f"global_advd"], D_losses[f"scat_advd"] = self.dualadv_loss.D(self.netD, comp_img, images, 1-masks)
            self.optimD.zero_grad()
            sum(D_losses.values()).backward()         
            self.optimD.step()

            # optimize G
            # reconstruction loss
            for name, weight in self.args.rec_loss.items(): 
                G_losses[name] = weight * self.rec_loss_func[name](pred_img, images)

            # dual adversarial loss
            G_losses[f"global_advg"], G_losses[f"scat_advg"] = self.dualadv_loss.G(self.netD, comp_img, 1-masks)
            G_losses[f"global_advg"], G_losses[f"scat_advg"] = self.args.adv_weight * G_losses[f"global_advg"], self.args.adv_weight * G_losses[f"scat_advg"]

            # contrastive learning losses
            textural_loss, semantic_loss = self.contrastive_loss(images_masked, comp_img, images, masks, 5)
            G_losses["contra_tex"] = self.args.text_weight * textural_loss
            G_losses["contra_sem"] = self.args.sem_weight * semantic_loss

            # netG and mlp backforward
            self.optimG.zero_grad()
            sum(G_losses.values()).backward()
            self.optimG.step()

            for name, value in D_losses.items():
                losses[name] = value
            for name, value in G_losses.items():
                losses[name] = value                


            if self.args.global_rank == 0:
                timer_model.hold()
                timer_data.tic()

            # logs
            scalar_reduced = reduce_loss_dict(losses, self.args.world_size)
            if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0): 
                pbar.update(self.args.print_every)
                description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                    if self.args.tensorboard: 
                        self.writer.add_scalar(key, val.item(), self.iteration)
                pbar.set_description((description))
                if self.args.tensorboard: 
                    self.writer.add_image('mask', make_grid(masks), self.iteration)
                    self.writer.add_image('orig', make_grid((images+1.0)/2.0), self.iteration)
                    self.writer.add_image('comp', make_grid((comp_img+1.0)/2.0), self.iteration)
                    
            
            if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0: 
                self.save()