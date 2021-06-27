from models.models import create_model
from options.test_options import TestOptions
import os
import torch
from torch.utils import data
import time
import numpy as np
import cv2
import tqdm
from data.TransferDataset import TransferDataset
from data.CarlaDataset import CarlaDataset

"""
--phase
test
--gpu_ids
0
--eval
--no_flip
--netG
Coarse2fineNet
--mode
Coarse2fine
--name
Coarse2fineNet_unet8_1206
--which_epoch
42
"""

"""
--phase
test
--gpu_ids
0
--eval
--no_flip
--netG
unet_256
--mode
Coarse
--name
CoarseNet_unet8_load400
--which_epoch
21
"""

"""
--phase
val
--gpu_ids
0
--eval
--no_flip
--netG
Coarse2fineNet
--mode
Transfer
--name
transferModel_0614
--which_epoch
42
"""



def test_one_epoch():
    opt = TestOptions().parse()
    model = create_model(opt)
    model.load_networks(opt.which_epoch)
    if opt.eval:
        model.netG.eval()
    dataset = TransferDataset(opt) if opt.mode == 'Transfer' else CarlaDataset(opt)
    iterator_test = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    with torch.no_grad():
        start_time = time.time()
        for COUNT, sample in tqdm.tqdm(enumerate(iterator_test)):
            visuals, error = model.test(sample)
            # save_visuals(visuals, opt, COUNT)
        print('time per sample', (time.time()-start_time) / dataset.__len__())


# def save_visuals(visuals, opt, COUNT):
#     name = visuals['inpaint_name']
#     # name = visuals['pic_name'] + '#'
#     des_dir = os.path.join(opt.results_dir, opt.name, opt.phase, 'epoch_'+str(opt.epoch))
#     if not os.path.exists(des_dir):
#         os.makedirs(des_dir)
#     inpaint_A = ((visuals['inpaint_A'] + 1) / 2. * 255).cpu().numpy()
#     # inpaint_B = ((visuals['inpaint_B'] + 1) / 2. * 255).cpu().numpy()
#     inpaint_C = ((visuals['inpaint_C']) * 255).cpu().numpy()
#     inpaint_fake_B = ((visuals['inpaint_fake_B'] + 1) / 2. * 255).cpu().numpy()
#     inpaint_fine_mask = ((visuals['inpaint_fine_mask']) * 255).cpu().numpy()
#     inpainting = np.concatenate((inpaint_C, inpaint_fine_mask, inpaint_A, inpaint_fake_B), 1)
#     cv2.imwrite(os.path.join(des_dir, name + '.png'), inpainting)

if __name__ == '__main__':
    test_one_epoch()