import os
import os.path as op
import random
import shutil
import time

import numpy as np
import torch

from datasets import build_dataloader
from model import build_model
from processor.processor import do_train
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_args(defaults={
        "num_epoch": 12,
        "batch_size": 48,
        "test_batch_size": 64,
        "img_aug": True,
    })
    if not torch.cuda.is_available():
        raise RuntimeError("Herb-LEGA with the frozen DALL-E 2 prior requires a CUDA GPU.")

    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        args.device = f"cuda:{args.local_rank}"
    else:
        args.device = "cuda:0"
        torch.cuda.set_device(0)

    set_seed(args.seed + get_rank())

    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f"{cur_time}_{args.name}")
    logger = setup_logger(
        "Herb-LEGA",
        save_dir=args.output_dir,
        if_train=True,
        distributed_rank=get_rank(),
    )
    logger.info("Using %d GPU(s)", num_gpus)
    logger.info(str(args).replace(",", "\n"))
    if get_rank() == 0:
        save_train_configs(args.output_dir, args)

    (
        train_loader,
        val_img_loader,
        val_txt_loader,
        test_img_loader,
        test_txt_loader,
        num_classes,
    ) = build_dataloader(args)

    model = build_model(args, num_classes).to(args.device)
    logger.info(
        "Trainable params: %.2fM | total params including frozen prior: %.2fM",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    checkpointer = Checkpointer(
        model,
        optimizer,
        scheduler,
        args.output_dir,
        save_to_disk=get_rank() == 0,
        logger=logger,
    )
    val_evaluator = Evaluator(
        val_img_loader,
        val_txt_loader,
        output_dir=args.output_dir,
        split_name="val",
    )

    start_epoch = 1
    best_instance_mr = float("-inf")
    best_epoch = 0
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        # This project saves best checkpoints with their validation score.
        best_instance_mr = float(checkpoint["best_instance_mr"])
        best_epoch = int(checkpoint["epoch"])
        if get_rank() == 0:
            best_path = op.join(args.output_dir, "best.pth")
            if not op.exists(best_path) or not op.samefile(args.resume_ckpt_file, best_path):
                shutil.copy2(args.resume_ckpt_file, best_path)
        synchronize()
        logger.info(
            "Resuming from epoch %d; best validation INSTANCE mR %.3f at epoch %d",
            start_epoch, best_instance_mr, best_epoch,
        )

    do_train(
        start_epoch,
        args,
        model,
        train_loader,
        val_evaluator,
        optimizer,
        scheduler,
        checkpointer,
        best_instance_mr=best_instance_mr,
        best_epoch=best_epoch,
    )

    synchronize()
    if get_rank() == 0 and args.test_after_train:
        best_path = op.join(args.output_dir, "best.pth")
        checkpointer.load(best_path)
        logger.info("Test Results from best validation INSTANCE-mR checkpoint")
        test_evaluator = Evaluator(
            test_img_loader,
            test_txt_loader,
            output_dir=args.output_dir,
            split_name="test",
        )
        test_evaluator.eval(model, epoch=None)
    synchronize()


if __name__ == "__main__":
    main()
