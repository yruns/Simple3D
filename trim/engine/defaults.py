"""
Default engine/testing logic
modified from Pointcept(https://github.com/Pointcept/Pointcept)

Please cite our work if the code is helpful to you.
"""

import multiprocessing as mp
import os
import shutil
from enum import Enum
from os.path import join

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trim.utils import dist, comm


class Mode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def wrap_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel models if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression callbacks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if dist.get_world_size() == 1:
        return model.cuda()
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [dist.get_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [dist.get_rank()]
    ddp = DistributedDataParallel(model.cuda(), **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def wrap_ddp_loader(dataset, batch_size, workers=0, collate_fn=None, drop_last=False, shuffle=None, **kwargs):
    if dist.get_world_size() > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    assert batch_size % dist.get_world_size() == 0, "Batch size should be divided by world size."
    batch_size = batch_size // dist.get_world_size()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) if shuffle is None else shuffle,
        num_workers=workers,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=True,
        **kwargs
    )

    return loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    comm.seed_everything(worker_seed)


def default_setup(cfg):
    # scalar by world size
    world_size = dist.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    # settle random seed
    rank = dist.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    comm.seed_everything(seed)
    return cfg


def save_checkpoint(state, is_best, output_dir, filename='model_last.pth'):
    filename = join(output_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(output_dir, 'model_best.pth'))


def save_checkpoint_epoch(state, output_dir, epoch):
    filename = join(output_dir, f'model_epoch_{epoch}.pth')
    torch.save(state, filename)


def load_state_dict(state_dict, model, logger, strict=True):
    try:
        load_state_info = model.load_state_dict(state_dict, strict=strict)
    except Exception:
        # The model was trained in a parallel manner, so need to be loaded differently
        from collections import OrderedDict
        weight = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # remove module
                k = k[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                # add module
                k = 'module.' + k  # xxx.xxx -> module.xxx.xxx
            weight[k] = v
        load_state_info = model.load_state_dict(weight, strict=strict)
    logger.info(f"Missing keys: {load_state_info[0]}")

    return model


def resume(weight, model, optimizer, scheduler, scaler, logger, strict=True):
    assert os.path.exists(weight), f"{weight} does not exist."

    logger.info("=> Loading checkpoint & weight at: {weight}")
    checkpoint = torch.load(
        weight,
        map_location=lambda storage, loc: storage.cuda(),
    )

    model = load_state_dict(checkpoint["state_dict"], model, logger, strict)
    logger.info(
        f"Resuming train at eval epoch: {checkpoint['epoch']}"
    )
    start_epoch = checkpoint["epoch"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    return start_epoch, model, optimizer, scheduler, scaler
