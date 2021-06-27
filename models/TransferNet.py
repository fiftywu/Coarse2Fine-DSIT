import torch
import random
import torch.nn.functional as F
from .base_model import BaseModel
from .loss import VGG16, PerceptualLoss
from . import networks

"""
--gpu_ids
0
--batchSize
1
--lr
0.0001
--netG
Coarse2fineNet
--netD
SA
--mode
Transfer
--name
transferModel_0614
--continue_train
--which_epoch
42
--epoch_count
43
"""

class TransferNet(BaseModel):
    def __init__(self, opt):
        super(TransferNet, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.vgg = VGG16()
        self.PerceptualLoss = PerceptualLoss()
        self.criterionL1 = torch.nn.L1Loss()
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
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

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
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
            # netG_param = netG_fine_net_param
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(netG_param, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isTrain:
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if self.isTrain:
            if self.opt.continue_train:
                self.load_networks(self.opt.which_epoch)


    def set_input(self, samples):
        def concat_synthesis(synthesis_data, item):
            tmp = []
            for each_dict in synthesis_data:
                tmp.append(each_dict[item])
            return torch.cat(tmp, 0)
        real_data, synthesis_data = samples
        real_data = real_data[0]
        self.inpaint_name = real_data['inpaint_name']
        if self.isTrain:
            # realistic
            self.inpaint_A = real_data['inpaint_A'].to(self.device)  # [-1,1]
            self.inpaint_B = real_data['inpaint_B'].to(self.device)  # [-1,1]
            self.inpaint_C = real_data['inpaint_C'].to(self.device)  # [0,1]

            # synthsis
            self.synt_A = concat_synthesis(synthesis_data, 'synt_A').to(self.device) # [-1,1]
            self.synt_B = concat_synthesis(synthesis_data, 'synt_B').to(self.device) # [-1,1]
            self.synt_C = concat_synthesis(synthesis_data, 'synt_C').to(self.device) # [0,1]
        else:
            # realistic
            self.inpaint_A = real_data['inpaint_A'].to(self.device)  # [-1,1]
            self.inpaint_B = real_data['inpaint_B'].to(self.device)  # [-1,1]
            self.inpaint_C = real_data['inpaint_C'].to(self.device)  # [0,1]
        # 0 is background; 1 is dynamic objects

        # plt.imshow(mask_new[0,:,:,:].view(256,256).cpu().detach().numpy(), cmap='gray')
        # plt.show()

    def forward(self):
        if self.isTrain:
            # synthesis
            synt_AC = torch.cat((self.synt_A, self.synt_C), 1)
            self.synt_fake_B, self.diff_mask, self.fine_mask, self.coarse_fake_B, self.fine_fake_B = self.netG(synt_AC)
        #realistic
        inpaint_AC = torch.cat((self.inpaint_A, self.inpaint_C), 1)
        self.inpaint_fake_B, self.inpaint_diff_mask, self.inpaint_fine_mask, self.inpaint_coarse_fake_B, self.inpaint_fine_fake_B = self.netG(inpaint_AC)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # D(x,m,y)
        # Fake; stop backprop to the generator by detaching fake_B
        all_real_A = torch.cat((self.synt_A, self.inpaint_A), 0)
        all_real_B = torch.cat((self.synt_B, self.inpaint_B), 0)
        all_real_C = torch.cat((self.fine_mask, self.inpaint_C))
        all_fake_B = torch.cat((self.synt_fake_B, self.inpaint_fake_B), 0)
        ones = torch.ones_like(all_real_A)

        fake_ABC = torch.cat((all_real_A, all_fake_B, all_real_C, ones), 1)
        pred_fake = self.netD(fake_ABC.detach())

        # Real
        real_ABC = torch.cat((all_real_A, all_real_B, all_real_C, ones), 1)
        pred_real = self.netD(real_ABC)

        # # # LSGAN
        self.loss_D_fake = torch.mean((pred_real-(0.9+random.random()*0.1))**2)
        self.loss_D_real = torch.mean((pred_fake-(0.0+random.random()*0.1))**2)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # completed imgs
        # First, GAN LOSS
        all_real_A = torch.cat((self.synt_A, self.inpaint_A), 0)
        all_real_B = torch.cat((self.synt_B, self.inpaint_B), 0)
        all_real_C = torch.cat((self.fine_mask, self.inpaint_C))
        all_fake_B = torch.cat((self.synt_fake_B, self.inpaint_fake_B), 0)
        ones = torch.ones_like(all_real_A)

        fake_ABC = torch.cat((all_real_A, all_fake_B, all_real_C, ones), 1)
        pred_fake = self.netD(fake_ABC)

        # # # LSGAN
        self.loss_G_GAN = torch.mean((pred_fake-1.)**2)

        # Other Losses
        self.loss_G_L1 = torch.sum(torch.abs(all_fake_B*all_real_C -
                                             all_real_B*all_real_C)) / (torch.sum(all_real_C) + 1) * 40
        self.loss_G_Percept = self.PerceptualLoss(
            torch.cat([all_fake_B, all_fake_B, all_fake_B], 1),
            torch.cat([all_real_B, all_real_B, all_real_B], 1),
            all_real_C)*0.2

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
        # loss['D_real'] = self.loss_D_real.item()
        # loss['D_fake'] = self.loss_D_fake.item()
        loss['G_percept'] = self.loss_G_Percept.item()
        return loss


    def get_current_visuals(self):
        visuals = dict()
        visuals['inpaint_A'] = self.inpaint_A[0][0]
        visuals['inpaint_B'] = self.inpaint_B[0][0]
        visuals['inpaint_C'] = self.inpaint_C[0][0]
        visuals['inpaint_fake_B'] = self.inpaint_fake_B[0][0]
        visuals['inpaint_fine_mask'] = self.inpaint_fine_mask[0][0]

        if self.isTrain:
            visuals['synt_A'] = self.synt_A[0][0]
            visuals['synt_B'] = self.synt_B[0][0]
            visuals['synt_C'] = self.synt_C[0][0]
            visuals['synt_fake_B'] = self.synt_fake_B[0][0]

        visuals['inpaint_name'] = self.inpaint_name[0]
        return visuals

    def get_statistic_errors(self):
        error = dict()
        inpaint_A = (self.inpaint_A[0][0] + 1)*0.5
        inpaint_B = (self.inpaint_A[0][0] + 1)*0.5
        inpaint_C = self.inpaint_A[0][0]
        error['L1'] = torch.sum(torch.abs(inpaint_A*-inpaint_B)) / (torch.sum(inpaint_C) + torch.sum(1-inpaint_C))
        error['L1_dmask'] = torch.sum(torch.abs(inpaint_A*inpaint_C-inpaint_B*inpaint_C)) / torch.sum(inpaint_C)
        error['L1_nodmask'] = torch.sum(torch.abs(inpaint_A*(1-inpaint_C)-inpaint_B*(1-inpaint_C))) / torch.sum(1-inpaint_C)
        error['DynaRate'] = torch.sum(inpaint_C) / (torch.sum(inpaint_C) + torch.sum(1-inpaint_C))
        return error

    def test(self, sample):
        self.set_input(sample)
        self.forward()
        visuals = self.get_current_visuals()
        error = self.get_statistic_errors()
        return visuals, error
