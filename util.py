import os
import torch
import random
import numpy as np

def smooth_label(tensor, offset):
    return tensor + offset

def time_output(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02dh:%02dm:%02ds" % (hour, min, sec)

def save_checkpoint(points, path, name='checkpoint.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(points, os.path.join(path, name))

def rotate(images):
    x = images
    x_90 = x.transpose(2,3)
    x_180 = x.flip(2,3)
    x_270 = x.transpose(2,3).flip(2,3)
    images = torch.cat((x, x_90, x_180, x_270), 0)
    return images

def seed_torch(seed=1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')