import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class GANLoss(torch.nn.Module):
    def __init__(self, target_real_label=1., target_fake_label=0.):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label


    def __call__(self, real_A, real_B, fake_B, dmask):
        pass
    # return errG

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()


        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()


        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])


        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)


        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)


        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3': max_3,


            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,


            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=None):
        super(PerceptualLoss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.add_module('vgg', VGG16().cuda())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y, dm):
        # Compute features
        x_vgg, y_vgg = self.vgg(x*dm), self.vgg(y*dm)
        content_loss = 0.0
        content_loss += self.weights[0] * partial_loss(x_vgg['relu1_1'], y_vgg['relu1_1'], dm)
        content_loss += self.weights[1] * partial_loss(x_vgg['relu2_1'], y_vgg['relu2_1'], dm)
        content_loss += self.weights[2] * partial_loss(x_vgg['relu3_1'], y_vgg['relu3_1'], dm)
        content_loss += self.weights[3] * partial_loss(x_vgg['relu4_1'], y_vgg['relu4_1'], dm)
        content_loss += self.weights[4] * partial_loss(x_vgg['relu5_1'], y_vgg['relu5_1'], dm)
        return content_loss/len(self.weights)

def partial_loss(x, y, m):
    ms = F.interpolate(m, size=(x.shape[2],x.shape[3]), mode='nearest')
    return torch.sum(torch.abs(x - y)) / x.shape[1] / (torch.sum(ms) + 1)