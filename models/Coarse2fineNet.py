import torch
import random
import torch.nn.functional as F
from .base_model import BaseModel
from .loss import VGG16, PerceptualLoss
from . import networks


'''
--gpu_ids
0
--batchSize
4
--netG
Coarse2fineNet
--netD
SA
--mode
Coarse2fine
--name
Coarse2fineNet_0627
'''


class Coarse2fineNet(BaseModel):
    def __init__(self, opt):
        super(Coarse2fineNet, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.vgg = VGG16()
        self.PerceptualLoss = PerceptualLoss()
        self.criterionL1 = torch.nn.L1Loss()
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define a generator
        '''
        input: 1+1: real_A + dmask
        '''
        self.netG = networks.define_G(1+1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        '''
        load the param for the coarse_net
        '''
        coarseNet_param = torch.load(self.opt.coarsenet_parms,
                                     map_location=lambda storage, loc: storage.cuda(opt.gpu_ids[0]))
        for params_name, params in self.netG.module.coarse_net.named_parameters():
            if params_name in self.netG.module.coarse_net.state_dict():
                self.netG.module.coarse_net.state_dict()[params_name].copy_(coarseNet_param['net'][params_name])

        # define a discriminator
        '''
        1+1+1 real_A, real_B, dmask, ones
        '''
        if self.isTrain:
            self.netD = networks.define_D(1+1+1+1, opt.ndf//2, opt.netD, opt.n_layers_D, opt.norm,
                                          opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.optimizers = []
            netG_coarse_net_param = self.netG.module.coarse_net.parameters()
            netG_fine_net_param = self.netG.module.fine_net.parameters()
            netG_param = [{'params': netG_coarse_net_param, 'lr': opt.lr*0.05},
                          {'params': netG_fine_net_param, 'lr': opt.lr}]

            self.optimizer_G = torch.optim.Adam(netG_param, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isTrain:
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, samples):
        self.real_A = samples['A'].to(self.device)
        self.real_B = samples['B'].to(self.device)
        self.pic_name = samples['output_name']
        # real semantic segmantation mask (pedestrain, vihicle)
        self.dmask = (samples['C'].to(self.device) > 0).float()  # 0 or 1
        self.ones = torch.ones_like(self.real_A)

    def forward(self):
        real_A_dmask = torch.cat((self.real_A, self.dmask), 1)
        self.fake_B, self.diff_mask, self.fine_mask, self.coarse_fake_B, self.fine_fake_B = self.netG(real_A_dmask)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_ABC = torch.cat((self.real_A, self.fake_B, self.fine_mask, self.ones), 1)
        pred_fake = self.netD(fake_ABC.detach())

        # Real
        real_ABC = torch.cat((self.real_A, self.real_B, self.fine_mask, self.ones), 1)
        pred_real = self.netD(real_ABC)

        # # # LSGAN
        self.loss_D_fake = torch.mean((pred_real-(0.9+random.random()*0.1))**2)
        self.loss_D_real = torch.mean((pred_fake-(0.0+random.random()*0.1))**2)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        # completed imgs
        # First, GAN LOSS
        fake_ABC = torch.cat((self.real_A, self.fake_B, self.fine_mask, self.ones), 1)
        pred_fake = self.netD(fake_ABC)

        # # # LSGAN
        self.loss_G_GAN = torch.mean((pred_fake-1.)**2)

        # Other Losses
        self.loss_G_L1 = torch.sum(torch.abs(self.fake_B*self.fine_mask -
                                             self.real_B*self.fine_mask)) / (torch.sum(self.fine_mask) + 1)*40
        self.loss_G_Percept = self.PerceptualLoss(
            torch.cat([self.fake_B, self.fake_B, self.fake_B], 1),
            torch.cat([self.real_B, self.real_B, self.real_B], 1),
            self.fine_mask)*0.2

        self.loss_G = self.loss_G_L1 + self.loss_G_GAN + self.loss_G_Percept
        self.loss_G.backward()

    def optimize_parameters(self):
        # compute fake images: G(A)
        self.forward()
        # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def get_current_loss(self):
        loss = dict()
        loss['G'] = self.loss_G.item()
        loss['G_GAN'] = self.loss_G_GAN.item()
        loss['G_L1'] = self.loss_G_L1.item()
        loss['D'] = self.loss_D.item()
        loss['D_real'] = self.loss_D_real.item()
        loss['D_fake'] = self.loss_D_fake.item()
        loss['G_percept'] = self.loss_G_Percept.item()
        return loss


    def get_current_visuals(self):
        visuals = dict()
        visuals['real_A'] = self.real_A[0][0]
        visuals['real_B'] = self.real_B[0][0]
        visuals['fake_B'] = self.fake_B[0][0]
        visuals['dmask'] = self.dmask[0][0] * 2 - 1  # -1 or 1
        visuals['diff_mask'] = self.diff_mask[0][0] * 2 - 1 # -1 or 1
        visuals['fine_mask'] = self.fine_mask[0][0] * 2 - 1 # -1 or 1
        visuals['coarse_fake_B'] = self.coarse_fake_B[0][0]
        visuals['fine_fake_B'] = self.fine_fake_B[0][0]
        visuals['pic_name'] = self.pic_name[0]
        return visuals

    def get_statistic_errors(self):
        error = dict()
        # from -1~1 to 0~1
        real_B = (self.real_B + 1) * 0.5
        fake_B = (self.fake_B + 1) * 0.5
        is_dmask = (self.dmask[0][0] >= 0.5).float()
        not_dmask = (self.dmask[0][0] < 0.5).float()
        error['L1'] = self.criterionL1(real_B, fake_B)
        dmask_count = torch.sum(is_dmask, dtype=torch.float) + 1.
        nodmask_count = torch.sum(not_dmask, dtype=torch.float) + 1.
        error['L1_dmask'] = torch.sum(torch.abs(real_B*is_dmask-fake_B*is_dmask)) / dmask_count
        error['L1_nodmask'] = torch.sum(torch.abs(real_B*not_dmask-fake_B*not_dmask)) / nodmask_count
        error['DynaRate'] = (dmask_count-1)/(dmask_count+nodmask_count - 2)
        return error

    def test(self, sample):
        self.set_input(sample)
        self.forward()
        visuals = self.get_current_visuals()
        error = self.get_statistic_errors()
        return visuals, error