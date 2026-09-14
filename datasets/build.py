import logging

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, ImageTextDataset, ImageTextMLMDataset, TextDataset
from .Herb163CMR import Herb163CMR


__factory = {"herb163cmr": Herb163CMR}


def build_transforms(img_size=(224, 224), aug=False, is_train=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        return T.Compose(
            [
                T.Resize((height, width)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    if aug:
        return T.Compose(
            [
                T.Resize((height, width)),
                T.RandomHorizontalFlip(0.5),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                T.RandomErasing(scale=(0.02, 0.4), value=mean),
            ]
        )

    return T.Compose(
        [
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def collate(batch):
    keys = set(key for item in batch for key in item.keys())
    output = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], int):
            output[key] = torch.tensor(values, dtype=torch.long)
        elif torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        else:
            raise TypeError(f"Unexpected data type for key {key}: {type(values[0])}")
    return output


def _make_eval_loaders(ds, transform, args):
    image_set = ImageDataset(
        image_ids=ds.get("image_ids", ds["image_pids"]),
        image_pids=ds["image_pids"],
        img_paths=ds["img_paths"],
        transform=transform,
    )
    text_set = TextDataset(
        caption_image_ids=ds.get("caption_image_ids", ds["caption_pids"]),
        caption_pids=ds["caption_pids"],
        captions=ds["captions"],
        caption_indices=ds.get("caption_indices"),
        text_length=args.text_length,
    )
    loader_kwargs = dict(
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return DataLoader(image_set, **loader_kwargs), DataLoader(text_set, **loader_kwargs)


def build_dataloader(args, transforms=None):
    logger = logging.getLogger("Herb-LEGA.dataset")

    if args.dataset_name not in __factory:
        raise KeyError(f"Unknown dataset {args.dataset_name}. Available: {sorted(__factory)}")

    if args.dataset_name == "herb163cmr":
        dataset = Herb163CMR(
            root=args.root_dir,
            source_jsonl=args.source_jsonl,
            split_dir=args.split_dir,
            strict=True,
            verbose=True,
        )

    num_classes = len(dataset.train_id_container)
    eval_transform = transforms or build_transforms(args.img_size, is_train=False)

    if not args.training:
        test_img_loader, test_txt_loader = _make_eval_loaders(dataset.test, eval_transform, args)
        return test_img_loader, test_txt_loader, num_classes

    train_transform = build_transforms(args.img_size, aug=args.img_aug, is_train=True)
    train_set_cls = ImageTextMLMDataset if args.MLM else ImageTextDataset
    train_set = train_set_cls(dataset.train, train_transform, text_length=args.text_length)

    sampler = None
    shuffle = True
    if args.distributed:
        sampler = DistributedSampler(train_set, shuffle=True)
        shuffle = False
        logger.info("Using DistributedSampler for training")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    val_img_loader, val_txt_loader = _make_eval_loaders(dataset.val, eval_transform, args)
    test_img_loader, test_txt_loader = _make_eval_loaders(dataset.test, eval_transform, args)
    return (
        train_loader,
        val_img_loader,
        val_txt_loader,
        test_img_loader,
        test_txt_loader,
        num_classes,
    )
