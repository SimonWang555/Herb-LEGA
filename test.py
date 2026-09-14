import os
import os.path as op
import random
import time

import numpy as np
import torch

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = get_args(defaults={"test_batch_size": 64})
    args.training = False
    args.distributed = False
    args.device = "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError("Herb-LEGA testing with GFA requires a CUDA GPU.")
    torch.cuda.set_device(0)
    set_seed(args.seed)

    checkpoint_path = args.checkpoint or args.resume_ckpt_file
    if not checkpoint_path:
        raise ValueError("Pass the trained checkpoint with --checkpoint /path/to/best.pth")

    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    test_output = op.join(args.output_dir, args.dataset_name, f"{cur_time}_{args.name}_test")
    logger = setup_logger("Herb-LEGA", test_output, if_train=False, distributed_rank=0)

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes).to(args.device)
    checkpointer = Checkpointer(model, save_dir=test_output, logger=logger)
    checkpointer.load(checkpoint_path)

    evaluator = Evaluator(
        test_img_loader,
        test_txt_loader,
        output_dir=test_output,
        split_name="test",
    )
    evaluator.eval(model, epoch=None)


if __name__ == "__main__":
    main()
