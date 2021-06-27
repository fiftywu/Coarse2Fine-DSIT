import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import os
from models.models import create_model
from options.train_options import TrainOptions
from data.TransferDataset import TransferDataset
from data.CarlaDataset import CarlaDataset


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = TransferDataset(opt) if opt.mode == 'Transfer' else CarlaDataset(opt)
    iterator_train = data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers)

    # Create model
    model = create_model(opt)
    model.print_networks()
    total_steps = 0

    # Create the logs
    log_dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir, comment=opt.name)

    # Start Training
    start_time = time.time()
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        for samples in iterator_train:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            model.set_input(samples)
            model.optimize_parameters()

            # display the training loss
            if total_steps % opt.print_freq == 0:
                loss = model.get_current_loss()
                for key in loss.keys():
                    writer.add_scalar(key, loss[key], total_steps + 1)
                print('epoch: ', epoch, 'total_steps: ', total_steps, 'loss: ', loss)

            # display the training processing
            if total_steps % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                # images = torch.cat((visuals['inpaint_A'],
                #                     visuals['inpaint_B'],
                #                     visuals['inpaint_C'],
                #                     visuals['inpaint_fake_B'],
                #                     visuals['inpaint_fine_mask'],
                #                     ), 1)
                # grid = torchvision.utils.make_grid(images)
                # writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps+1), grid, total_steps+1)
            if total_steps % opt.save_latest_freq == 0:
                model.save_networks(opt.which_epoch)
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)
        model.update_learning_rate()
    writer.close()