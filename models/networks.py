import os
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import cv2
from .gated_networks import GatedConv2dWithActivation, GatedDeConv2dWithActivation, SNConvWithActivation, get_pad
from .camodels import ContextualAttention

###############################################################################
# Helper Functions
###############################################################################
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- name of netG
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Coarse2fineNet':
        net = Coarse2fineNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        print('Not Found-->', netG)
        net = None

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70脳70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'SA':
        net = InpaintSADirciminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)



def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions锛庛€€
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Coarse2fineNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Coarse2fineNet, self).__init__()
        '''
        input: real_A + dmask
        output: coarse_fake_B
        '''
        self.coarse_net = UnetGenerator(1+1, 1, 8, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        '''
        input: coarse_image + dmask2
        output: fine_fake_B
        '''
        self.fine_net = InpaintSANet(1+1, 1, cnum=ngf//2, norm_layer=norm_layer, use_dropout=use_dropout)
        '''
        step1: -> real_A+dmask -> coarse_fake_B(whole)
        step2: -> coarse_fake_B+real_A+dmask -> fine_mask
        step3: -> fine_mask+coarse_fake_B+real_A -> fine_fake_B(whole)
        m2: float
        '''

    def forward(self, xm, tau=0.3):
        batch_size = xm.shape[0]
        real_A = xm[:, 0, :, :].view(batch_size, 1, 256, 256)
        dmask = xm[:, 1, :, :].view(batch_size, 1, 256, 256)
        coarse_fake_B = self.coarse_net(torch.cat([real_A, dmask], 1))
        diff_mask = get_diff_mask(coarse_fake_B, real_A, dmask, tau=tau)
        fine_mask = get_fine_mask(diff_mask, dmask)
        fine_fake_B = self.fine_net(torch.cat([real_A, coarse_fake_B, fine_mask], 1))
        fake_B = real_A*(1.-fine_mask) + fine_fake_B*fine_mask
        return fake_B, diff_mask, fine_mask, coarse_fake_B, fine_fake_B


# Define the resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)


class InpaintSANet(torch.nn.Module):
    """
    # Structure and Texture Inpainting
    Inpaint generator, input should be 3*256*256, where 1*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the ones
    """

    def __init__(self, in_channel=3, out_channel=1, cnum=32, norm_layer=nn.BatchNorm2d, use_dropout = False):
        super(InpaintSANet, self).__init__()
        activateFunc = nn.LeakyReLU(0.2, True)
        self.pm1 = GatedConv2dWithActivation(in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1), activation=activateFunc, norm_layer=norm_layer)
        self.pm2_down = GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2), activation=activateFunc, norm_layer=norm_layer)
        self.pm3 = GatedConv2dWithActivation(2*cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activateFunc, norm_layer=norm_layer)  # output 128
        self.pm4_down = GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2), activation=activateFunc, norm_layer=norm_layer)
        self.pm5 = GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activateFunc, norm_layer=norm_layer)  # input 256 output 128

        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)
        self.contextul_attention_an = nn.Sequential(activateFunc, norm_layer(4*cnum))
        self.con1_down = nn.Sequential(nn.Conv2d(5*cnum, 4*cnum, kernel_size=3, stride=2, padding=1), activateFunc, norm_layer(4*cnum))
        # self.con1_down = GatedConv2dWithActivation(5 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2), activation=activateFunc, norm_layer=norm_layer)
        # self.con2 = GatedConv2dWithActivation(12 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activateFunc, norm_layer=norm_layer)
        self.con2 = nn.Sequential(nn.Conv2d(12*cnum, 4*cnum, kernel_size=3, stride=1, padding=1), activateFunc, norm_layer(4*cnum))

        # st
        self.st1 = nn.Sequential(nn.Conv2d(8*cnum, 4*cnum, kernel_size=3, stride=1, padding=1), activateFunc, norm_layer(4*cnum))
        self.st2 = nn.Sequential(nn.Conv2d(8*cnum, 4*cnum, kernel_size=3, stride=1, padding=1), activateFunc, norm_layer(4*cnum))
        self.st3 = nn.Sequential(nn.ConvTranspose2d(8*cnum, 2*cnum, kernel_size=4, stride=2, padding=1), activateFunc, norm_layer(2*cnum))
        self.st4 = nn.Sequential(nn.ConvTranspose2d(8*cnum, 2*cnum, kernel_size=4, stride=2, padding=1), activateFunc, norm_layer(2*cnum))
        self.st5 = nn.Sequential(nn.ConvTranspose2d(8*cnum, cnum, kernel_size=4, stride=4, padding=0), activateFunc, norm_layer(cnum))

        # st loss
        # self.ss = nn.Sequential(nn.ConvTranspose2d(4*cnum, 1, kernel_size=4, stride=4, padding=0), nn.Tanh())
        # self.tt = nn.Sequential(nn.ConvTranspose2d(4*cnum, 1, kernel_size=4, stride=4, padding=0), nn.Tanh())

        res_num = 4
        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(cnum * 4, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        # Decode convolution
        self.decode1 = GatedConv2dWithActivation(12 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activateFunc, norm_layer=norm_layer)
        self.decode2_up = GatedDeConv2dWithActivation(2, 12 * cnum, 2 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activateFunc, norm_layer=norm_layer)  # output 128
        self.decode3 = GatedConv2dWithActivation(6 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activateFunc, norm_layer=norm_layer)
        self.decode4_up = GatedDeConv2dWithActivation(2, 6 * cnum, 1 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activateFunc, norm_layer=norm_layer)  # output 256
        self.decode5 = GatedConv2dWithActivation(3 * cnum, out_channel, 3, 1, padding=get_pad(256, 3, 1), activation=activateFunc, norm_layer=norm_layer)
        self.activate = nn.Tanh()
        self.selayer = nn.Sequential(SELayer(8 * cnum, 16), norm_layer(8 * cnum))
        # self.selayer = nn.Sequential(SELayer(4 * cnum, 16), norm_layer(4 * cnum))

    def forward(self, xm):
        # [real_A, coarse_fake_B, fine_mask] ->-> [x, x_stage1, m2]
        # plt.imshow(Gx.view(256,256).cpu().detach().numpy(), cmap='gray')
        # plt.show()
        batch_size = xm.shape[0]
        x = xm[:, 0, :, :].view(batch_size, 1, 256, 256)
        x_stage1 = xm[:, 1, :, :].view(batch_size, 1, 256, 256)
        m2 = xm[:, 2, :, :].view(batch_size, 1, 256, 256)
        x_stage1_fill = x_stage1*m2 + x*(1-m2)

        p1 = self.pm1(torch.cat((x_stage1_fill, m2), 1))
        p2 = self.pm2_down(p1)
        p3 = self.pm3(p2)
        p4 = self.pm4_down(p3)
        p5 = self.pm5(p4)
        p6 = self.middle(p5)

        p1_down = F.interpolate(p1, size=(128, 128), mode='nearest')
        p123 = torch.cat((p1_down, p2, p3), 1)
        p123_ca = self.con1_down(p123)
        p123_ca, offset = self.contextul_attention(p123_ca, p123_ca, m2)

        p456 = torch.cat((p4, p5, p6), 1)
        p456 = self.con2(p456)
        p456_ca, _ = self.contextul_attention(p456, p456, m2)

        # decode with structure and texture
        st = torch.cat((p123_ca, p456_ca),1)
        st = self.selayer(st)
        dst1 = self.st1(st)
        dst2 = self.st2(st)
        dst3 = self.st3(st)
        dst4 = self.st4(st)
        dst5 = self.st5(st)
        # tt = self.tt(p123_ca)
        # ss = self.ss(p456_se)
        xx1 = self.decode1( torch.cat((dst1, p5, p6), 1) )
        xx2 = self.decode2_up( torch.cat((dst2, xx1, p4), 1) )
        xx3 = self.decode3( torch.cat((dst3, xx2, p3), 1) )
        xx4 = self.decode4_up( torch.cat((dst4, xx3, p2), 1) )
        xx5 = self.decode5( torch.cat((dst5, xx4, p1), 1) )
        x_stage2 = self.activate(xx5)
        return x_stage2


'''
used scripts of Coarse2fineNet
'''
def get_diff_mask(coarse_fake_B, real_A, dmask, tau=0.3):
    # input all float
    # output bool
    delta = torch.abs(coarse_fake_B-real_A)  # [0,2]
    diff_mask = delta > tau
    # diff_mask_final = diff_mask.float()
    # filter
    filter_diff_mask = F.avg_pool2d(diff_mask.float(), kernel_size=5, stride=1)
    filter_diff_mask = F.interpolate(filter_diff_mask, size=(256, 256), mode='nearest')
    diff_mask_final = (filter_diff_mask > 0.45).float()
    return diff_mask_final


def get_fine_mask(diff_mask, dmask):
    # input: diff_mask(bool), dmask(float)
    # output: fine_mask(float)
    batch_size = dmask.shape[0]
    fine_mask = torch.add(diff_mask > 0, dmask > 0).float()  # m2=1. is dynamic/ hole
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    for batch_item in range(batch_size):
        src = fine_mask[batch_item, 0, :, :].view(256, 256).cpu().detach().numpy()
        fine_mask[batch_item, 0, :, :] = torch.tensor(cv2.dilate(src, kernel)).to(dmask.device)
    return fine_mask


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class InpaintSADirciminator(nn.Module):
    def __init__(self, n_channel=4, cnum=32, norm_layer=nn.BatchNorm2d):
        super(InpaintSADirciminator, self).__init__()
        # cnum = 32
        # n_channel = 4
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(n_channel, 2*cnum, 4, 2, padding=get_pad(256, n_channel, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, n_channel, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, n_channel, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, n_channel, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, n_channel, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, n_channel, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, n_channel, 2))
        )

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view( (x.size(0), -1) )
        return x
