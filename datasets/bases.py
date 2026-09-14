from typing import Sequence

import logging
import random
import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset

from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer


class BaseDataset(object):
    logger = logging.getLogger("Herb-LEGA.dataset")

    def show_dataset_info(self):
        num_train_pids = len(self.train_id_container)
        num_train_imgs = len(self.train_annos)
        num_train_captions = len(self.train)
        num_test_pids = len(self.test_id_container)
        num_test_imgs = len(self.test_annos)
        num_test_captions = len(self.test["captions"])
        num_val_pids = len(self.val_id_container)
        num_val_imgs = len(self.val_annos)
        num_val_captions = len(self.val["captions"])

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(["subset", "ids", "images", "captions"])
        table.add_row(["train", num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(["val", num_val_pids, num_val_imgs, num_val_captions])
        table.add_row(["test", num_test_pids, num_test_imgs, num_test_captions])
        self.logger.info("\n" + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if not truncate:
            raise RuntimeError(f"Input caption is too long for context length {text_length}")
        tokens = tokens[:text_length]
        tokens[-1] = eot_token
    result[: len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self, dataset, transform=None, text_length=77, truncate=True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if len(sample) == 5:
            pid, image_id, img_path, caption, caption_idx = sample
        elif len(sample) == 4:
            pid, image_id, img_path, caption = sample
            caption_idx = 0
        else:
            raise ValueError(f"Unexpected training sample format: {sample!r}")
        image = read_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
        caption_ids = tokenize(
            caption,
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return {
            "pids": int(pid),
            "image_ids": int(image_id),
            "caption_indices": int(caption_idx),
            "images": image,
            "caption_ids": caption_ids,
        }


class ImageDataset(Dataset):
    def __init__(self, image_ids, image_pids, img_paths, transform=None):
        self.image_ids = image_ids
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = read_image(self.img_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image_ids": torch.tensor(int(self.image_ids[index]), dtype=torch.long),
            "pids": torch.tensor(int(self.image_pids[index]), dtype=torch.long),
            "images": image,
        }


class TextDataset(Dataset):
    def __init__(
        self,
        caption_image_ids,
        caption_pids,
        captions,
        caption_indices=None,
        text_length=77,
        truncate=True,
    ):
        self.caption_image_ids = caption_image_ids
        self.caption_pids = caption_pids
        self.captions = captions
        self.caption_indices = caption_indices or [0] * len(captions)
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption_ids = tokenize(
            self.captions[index],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return {
            "image_ids": torch.tensor(int(self.caption_image_ids[index]), dtype=torch.long),
            "pids": torch.tensor(int(self.caption_pids[index]), dtype=torch.long),
            "caption_indices": torch.tensor(int(self.caption_indices[index]), dtype=torch.long),
            "caption_ids": caption_ids,
        }


class ImageTextMLMDataset(ImageTextDataset):
    def __getitem__(self, index):
        ret = super().__getitem__(index)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(
            ret["caption_ids"].cpu().numpy().copy()
        )
        ret["mlm_ids"] = mlm_tokens
        ret["mlm_labels"] = mlm_labels
        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        tokens[i] = mask
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)
                    labels.append(token)
                else:
                    labels.append(0)
            else:
                labels.append(0)
        if all(label == 0 for label in labels):
            labels[1] = tokens[1]
            tokens[1] = mask
        return torch.tensor(tokens), torch.tensor(labels)
