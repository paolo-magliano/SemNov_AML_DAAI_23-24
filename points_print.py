import numpy as np

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.modelnet import *
from datasets.scanobject import *
from classifiers.trainer_cla_md import *

import pyvista as pv

if __name__ == '__main__':
    opt = get_args()

    print(f"Arguments: {opt}")
    set_random_seed(opt.seed)

    dataloader_config = {
        'batch_size': opt.batch_size, 'drop_last': False, 'shuffle': False,
        'num_workers': opt.num_workers, 'sampler': None, 'worker_init_fn': init_np_seed}

    # whole evaluation is done on ScanObject RW data
    sonn_args = {
        'data_root': opt.data_root,
        'sonn_split': opt.sonn_split,
        'h5_file': opt.sonn_h5_name,
        'split': 'all',  # we use both training (unused) and test samples during evaluation
        'num_points': opt.num_points_test,  # default: use all 2048 sonn points to avoid sampling randomicity
        'transforms': None  # no augmentation applied at inference time
    }

    mn_loader, _ = get_md_eval_loaders(opt)

    sr1_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet1", **sonn_args), **dataloader_config)
    sr2_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet2", **sonn_args), **dataloader_config)
    sr3_loader = DataLoader(ScanObject(class_choice="sonn_ood_common", **sonn_args), **dataloader_config)

    loaders = {
        'MN': mn_loader,
        'SR1': sr1_loader,
        'SR2': sr2_loader,
        'SR3': sr3_loader
    }

    select = 'MN'
    loader = loaders[select]

    for i, batch in enumerate(loader, 0):
        points, labels = batch[0], batch[1]
        for point, label in zip(points, labels):
            print(f"Label: {label}, Points: {point.shape}")
            pv.plot(point)
            break
