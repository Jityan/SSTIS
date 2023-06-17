import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import time
from network import Generator, Discriminator, KL_loss
from util import time_output, save_checkpoint, rotate, seed_torch, smooth_label, str2bool

from cub_dataset import CUBTextDataset
from oxford_dataset import OxfordTextDataset


class Trainer(object):
    def __init__(self, args):
        self.noise_dim = 100
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.beta1 = 0.5
        self.num_epochs = args.epochs
        self.checkpoints_path = args.exp_num
        self.save_path = args.save_path
        # models
        self.gen = Generator().cuda()
        self.disc = Discriminator().cuda()
        # optimizer
        self.optimD = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # scheduler
        self.dScheduler = torch.optim.lr_scheduler.StepLR(self.optimD, step_size=args.lr_decay_epoch, gamma=args.lr_decay_step)
        self.gScheduler = torch.optim.lr_scheduler.StepLR(self.optimG, step_size=args.lr_decay_epoch, gamma=args.lr_decay_step)

        if args.dataset == 'birds':
            print("=> CUB {} dataset...".format(args.split))
            self.dataset = CUBTextDataset(split=args.split, imsize=args.imsize)
        else:
            print("=> Oxford {} dataset...".format(args.split))
            self.dataset = OxfordTextDataset(split=args.split, imsize=args.imsize)
        #
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.args = args

    def train(self):
        start_epoch = 0
        # prepare metrices
        trlog = {
            'args': self.args,
            'hist_d': [],
            'hist_dr': [],
            'hist_df': [],
            'hist_g': [],
            'hist_gb': [],
        }
        # check resume point
        checkpoint_file = os.path.join(self.save_path, self.checkpoints_path, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            trlog = checkpoint['trlog']
            start_epoch = checkpoint['start_epoch'] + 1
            self.disc.load_state_dict(checkpoint['netD_state_dict'])
            self.gen.load_state_dict(checkpoint['netG_state_dict'])
            self.optimG.load_state_dict(checkpoint['optimG'])
            self.optimD.load_state_dict(checkpoint['optimD'])
            self.dScheduler.load_state_dict(checkpoint['dlr'])
            self.gScheduler.load_state_dict(checkpoint['glr'])
            print("Resume from epoch {} ...".format(start_epoch+1))
            for param_group in self.optimG.param_groups:
                print("=> Generator used learning rate :", param_group['lr'])
            for param_group in self.optimD.param_groups:
                print("=> Discriminator used learning rate :", param_group['lr'])

        criterion = nn.BCELoss().cuda()
        l2_loss = nn.MSELoss().cuda()
        l1_loss = nn.L1Loss().cuda()

        for epoch in range(start_epoch, self.num_epochs):
            time1 = time.time()
            self.disc.train()
            self.gen.train()
            # prepare metrics
            temp_log = {
                'd_loss': [],
                'd_rloss': [],
                'd_floss': [],
                'g_loss': [],
                'g_bloss': [],
            }

            for sample in self.data_loader:
                right_images = sample['right_images'].cuda()
                right_embed = sample['right_embed'].cuda()
                wrong_images = sample['wrong_images'].cuda()
                # generate image
                noise = torch.FloatTensor(right_images.size(0), self.noise_dim).cuda()
                noise.data.normal_(0,1)
                fake_images, mu, logvar = self.gen(right_embed, noise)
                
                # preprocess image into rotation form
                bs = right_images.size(0)
                right_images = rotate(right_images)
                wrong_images = rotate(wrong_images)
                fake_images = rotate(fake_images)
                mu_ext = mu.repeat(4, 1)
                # rot label
                rot_labels = torch.zeros(4*bs,).cuda()
                for i in range(4*bs):
                    if i < bs:
                        rot_labels[i] = 0
                    elif i < 2*bs:
                        rot_labels[i] = 1
                    elif i < 3*bs:
                        rot_labels[i] = 2
                    else:
                        rot_labels[i] = 3
                rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                real_labels = torch.ones(right_images.size(0))
                smoothed_real_labels = torch.FloatTensor(smooth_label(real_labels.numpy(), args.penalty)).cuda()
                real_labels = real_labels.cuda()
                fake_labels = torch.zeros(right_images.size(0)).cuda()
                # train discriminator
                self.disc.zero_grad()
                d_rloss, d_floss, d_sslloss = self.disc_loss(
                    criterion, right_images, wrong_images, fake_images, mu_ext, rot_labels, smoothed_real_labels, fake_labels)
                d_loss = d_rloss + d_floss + d_sslloss
                d_loss.backward()
                self.optimD.step()
                # train generator
                self.gen.zero_grad()
                g_loss, g_bloss = self.gen_loss(
                    criterion, l1_loss, l2_loss, fake_images, right_images, mu_ext, rot_labels, real_labels)
                g_loss += (args.ca_coef * KL_loss(mu, logvar))
                g_loss.backward()
                self.optimG.step()

                temp_log['d_loss'].append(d_loss.data.cpu().mean())
                temp_log['d_rloss'].append(d_rloss.data.cpu().mean())
                temp_log['d_floss'].append(d_floss.data.cpu().mean())
                temp_log['g_loss'].append(g_loss.data.cpu().mean())
                temp_log['g_bloss'].append(g_bloss.data.cpu().mean())
            # scheduler
            self.dScheduler.step()
            self.gScheduler.step()
            time2 = time.time()
            # update 1 epoch loss
            print("Epoch: {}/{}, d_loss={:.4f} - g_loss={:.4f} [{} total {}]".format(
                (epoch+1),
                self.num_epochs,
                np.array(temp_log['d_loss']).mean(),
                np.array(temp_log['g_loss']).mean(),
                datetime.datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).strftime("%H:%M"),
                time_output(time2-time1)
                )
            )

            trlog['hist_d'].append(np.array(temp_log['d_loss']).mean())
            trlog['hist_dr'].append(np.array(temp_log['d_rloss']).mean())
            trlog['hist_df'].append(np.array(temp_log['d_floss']).mean())
            trlog['hist_g'].append(np.array(temp_log['g_loss']).mean())
            trlog['hist_gb'].append(np.array(temp_log['g_bloss']).mean())

            temp_log['d_loss'] = []
            temp_log['d_rloss'] = []
            temp_log['d_floss'] = []
            temp_log['g_loss'] = []
            temp_log['g_bloss'] = []

            save_checkpoint({
                'start_epoch': epoch,
                'netG_state_dict': self.gen.state_dict(),
                'netD_state_dict': self.disc.state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict(),
                'dlr': self.dScheduler.state_dict(),
                'glr': self.gScheduler.state_dict(),
                'trlog': trlog
            }, os.path.join(self.save_path, self.checkpoints_path))
            #
            if (epoch+1) % 100 == 0:
                save_checkpoint({
                    'start_epoch': epoch,
                    'netG_state_dict': self.gen.state_dict(),
                }, os.path.join(self.save_path, self.checkpoints_path), name='epoch'+str(epoch+1)+'.pth.tar')

        #plot training graph
        plt.plot(trlog['hist_dr'], label='D real')
        plt.plot(trlog['hist_df'], label='D fake')
        plt.plot(trlog['hist_gb'], label='G base')
        plt.legend()
        plt.show()
    
    def disc_loss(self, criterion, right_images, wrong_images, fake_images, embed, rot_labels, smoothed_real_labels, fake_labels):
        embed = embed.detach()
        # obtain image feature
        rfeat, rot_logit = self.disc(right_images)
        wfeat, _ = self.disc(wrong_images)
        ffeat, _ = self.disc(fake_images.detach())
        # ssl loss
        rot_loss = torch.sum(F.binary_cross_entropy_with_logits(
            input=rot_logit,
            target=rot_labels
        ))
        # real condition loss
        r_cond_logit = self.disc.cond_output(rfeat, embed)
        real_cond_loss = criterion(r_cond_logit, smoothed_real_labels)
        # wrong condition loss
        w_cond_logit = self.disc.cond_output(wfeat, embed)
        wrong_cond_loss = criterion(w_cond_logit, fake_labels)
        # fake condition loss
        f_cond_logit = self.disc.cond_output(ffeat, embed)
        fake_cond_loss = criterion(f_cond_logit, fake_labels)
        # uncondition loss
        r_logits = self.disc.uncond_output(rfeat)
        real_loss = criterion(r_logits, smoothed_real_labels)
        f_logits = self.disc.uncond_output(ffeat)
        fake_loss = criterion(f_logits, fake_labels)
        # final disc loss
        d_rloss = (real_loss + real_cond_loss) / 2.0
        d_floss = (fake_loss + fake_cond_loss + wrong_cond_loss) / 3.0
        d_sslloss = (args.d_ssl * rot_loss)
        return d_rloss, d_floss, d_sslloss
    
    def gen_loss(self, criterion, l1_loss, l2_loss, fake_images, right_images, embed, rot_labels, real_labels):
        embed = embed.detach()
        # obtain image feature
        ffeat, f_rot_logit = self.disc(fake_images)
        rfeat, _ = self.disc(right_images)
        # ssl loss
        g_rot_fake_loss = torch.sum(F.binary_cross_entropy_with_logits(
            input=f_rot_logit,
            target=rot_labels
        ))
        # l2
        activation_fake = torch.mean(ffeat, 0)
        activation_real = torch.mean(rfeat, 0)
        # conditional loss
        g_cond_logit = self.disc.cond_output(ffeat, embed) 
        g_cond_loss = criterion(g_cond_logit, real_labels)
        # unconditional loss
        g_logits = self.disc.uncond_output(ffeat)
        g_uncond_loss = criterion(g_logits, real_labels)
        # final loss
        g_bloss = (g_cond_loss + g_uncond_loss)
        g_loss = g_bloss + (args.gamma * l1_loss(fake_images, right_images)) # L1 loss
        g_loss += (args.beta * l2_loss(activation_fake, activation_real.detach())) # feature matching loss
        g_loss += (args.g_ssl * g_rot_fake_loss)
        return g_loss, g_bloss

    def predict(self, target='checkpoint.pth.tar'):
        import re
        targetfilename = target
        tfn = targetfilename.split('.')[0]
        checkpoint_file = os.path.join(self.save_path, args.exp_num, targetfilename)
        if not os.path.isfile(checkpoint_file):
            print("Pretrained model not found...")
            return False
            
        checkpoint = torch.load(checkpoint_file)
        self.gen.load_state_dict(checkpoint['netG_state_dict'])
        print("Model loaded...")
        self.gen.eval()
        
        imgcount = 0
        fake_path = '{0}/{1}/fake_images_{2}'.format(self.save_path, self.checkpoints_path, tfn)
        if not os.path.exists(fake_path):
            os.makedirs(fake_path)
        
        for i in range(10):
            for sample in self.data_loader:
                right_embed = sample['right_embed'].cuda()
                txt = sample['txt']

                # generate fake images
                noise = torch.FloatTensor(right_embed.size(0), self.noise_dim).cuda()
                noise.data.normal_(0,1)
                fake_images, _, _ = self.gen(right_embed, noise)

                for fakeimg, t in zip(fake_images, txt):
                    fim = Image.fromarray(fakeimg.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    t = re.sub("[^0-9a-zA-Z]+", " ", t)
                    if len(t) > 100:
                        t = t[:100]
                    fim.save(os.path.join(fake_path, '{0}_{1}.jpg'.format(imgcount, t)))
                    imgcount += 1
        print("Complete... total image :", imgcount)

import pytz
def main(args):
    tz = pytz.timezone('Asia/Kuala_Lumpur')
    starttime = datetime.datetime.now(tz)
    print("=> Train start :", starttime)
    seed_torch(seed=args.seed)
    if args.is_test:
        args.split = 'test'
    else:
        args.split = 'train'
    trainer = Trainer(args)
    if not args.is_test:
        trainer.train()
    else:
        trainer.predict(target=args.target)
    print("=> Total executed time :", datetime.datetime.now(tz) - starttime)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text-to-image synthesis')
    parser.add_argument('--save_path', type=str, default='./saved_model')
    parser.add_argument('--exp_num', type=str, default="cub_exp")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--imsize', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--d_ssl', type=float, default=2.0)
    parser.add_argument('--g_ssl', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=int, default=100)
    parser.add_argument('--lr_decay_step', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--target', type=str, default='checkpoint.pth.tar')
    parser.add_argument('--penalty', type=float, default=-0.1)
    parser.add_argument('--ca_coef', type=float, default=5.0)
    parser.add_argument('--dataset', default='birds', choices=['birds','flowers'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--is_test', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    main(args)
