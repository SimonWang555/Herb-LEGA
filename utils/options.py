import argparse
import os


def get_args(defaults=None):
    parser = argparse.ArgumentParser(description="Herb-LEGA official implementation for Herb163CMR")

    # General
    parser.add_argument("--local_rank", default=int(os.getenv("LOCAL_RANK", 0)), type=int)
    parser.add_argument("--name", default="Herb-LEGA")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--log_period", default=50, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt_file", default="")
    parser.add_argument("--checkpoint", default="", help="checkpoint used by test.py")
    parser.add_argument("--test", dest="training", default=True, action="store_false")
    parser.add_argument("--no_test_after_train", dest="test_after_train", action="store_false")
    parser.set_defaults(test_after_train=True)

    # Model
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gen_temperature", type=float, default=0.07)
    parser.add_argument("--img_aug", action="store_true")
    parser.add_argument("--MLM", action="store_true")

    # Paper losses: loss_itc + alpha loss_clfe + beta loss_slfe + gamma loss_gfa
    parser.add_argument("--loss_names", default="itc+gfa+clfe+slfe")
    parser.add_argument("--top_weight", type=float, default=0.7, help="alpha")
    parser.add_argument("--pool_weight", type=float, default=0.8, help="beta")
    parser.add_argument("--pred_weight", type=float, default=0.03, help="gamma")

    # Vision and text
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    # Optimizer and paper schedule
    parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-3)
    parser.add_argument("--num_epoch", type=int, default=6)
    parser.add_argument("--milestones", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_method", default="linear", choices=["constant", "linear"])
    parser.add_argument("--lrscheduler", default="cosine", choices=["step", "exp", "poly", "cosine", "linear"])
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--lr_factor", type=float, default=5.0)

    # Herb163CMR fixed data protocol
    parser.add_argument("--dataset_name", default="herb163cmr", choices=["herb163cmr"])
    parser.add_argument("--sampler", default="random", choices=["random"])
    parser.add_argument("--root_dir", default="Herb163CMR")
    parser.add_argument(
        "--source_jsonl",
        default="Herb163CMR/Herb163CMR_Descriptions.jsonl",
    )
    parser.add_argument("--split_dir", default="Herb163CMR/splits/seed_42")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # LFE parameters from the paper
    parser.add_argument("--text_k", type=int, default=10)
    parser.add_argument("--image_k", type=int, default=10)
    parser.add_argument("--pool_k", type=int, default=3)

    # Frozen DALL-E 2 prior used by GFA
    parser.add_argument(
        "--prior_config",
        default="pretrained/dalle2_prior/prior_config.json",
    )
    parser.add_argument(
        "--prior_checkpoint",
        default="pretrained/dalle2_prior/best.pth",
    )
    parser.add_argument("--prior_sample_timesteps", type=int, default=1)

    # Entry-point defaults are applied before parsing so CLI values take priority.
    if defaults is not None:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    args.img_size = tuple(args.img_size)
    return args
