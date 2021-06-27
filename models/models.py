from .Coarse2fineNet import Coarse2fineNet
from .CoarseNet import CoarseNet
from .TransferNet import TransferNet

def create_model(opt):
    if opt.mode == 'Coarse':
        model = CoarseNet(opt)
    elif opt.mode == 'Coarse2fine':
        model = Coarse2fineNet(opt)
    elif opt.mode == 'Transfer':
        model = TransferNet(opt)
    else:
        print('NOT FOUND->', opt.mode)
        model = None
    return model
