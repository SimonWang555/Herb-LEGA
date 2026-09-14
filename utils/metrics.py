import json
import logging
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from prettytable import PrettyTable


def _recall_at_k(
    similarity: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks=(1, 5, 10),
) -> Dict[str, float]:
    if similarity.ndim != 2:
        raise ValueError(f"Expected a 2-D similarity matrix, got {similarity.shape}")
    if similarity.shape[0] != query_labels.numel():
        raise ValueError("Query label count does not match similarity rows.")
    if similarity.shape[1] != gallery_labels.numel():
        raise ValueError("Gallery label count does not match similarity columns.")

    max_k = min(max(ks), similarity.shape[1])
    indices = torch.topk(similarity, k=max_k, dim=1, largest=True, sorted=True).indices
    predicted = gallery_labels[indices]
    matches = predicted.eq(query_labels.view(-1, 1))

    recalls = {}
    for k in ks:
        effective_k = min(k, max_k)
        value = matches[:, :effective_k].any(dim=1).float().mean().item() * 100.0
        recalls[f"R@{k}"] = value
    return recalls


def _compute_level_metrics(
    similarity_t2i: torch.Tensor,
    text_labels: torch.Tensor,
    image_labels: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    t2i = _recall_at_k(similarity_t2i, text_labels, image_labels)
    i2t = _recall_at_k(similarity_t2i.t(), image_labels, text_labels)
    mr = sum(i2t.values()) / 3.0 / 2.0 + sum(t2i.values()) / 3.0 / 2.0
    return {"I2T": i2t, "T2I": t2i, "mR": mr}


def format_result_line(level: str, metrics: Dict[str, Dict[str, float]]) -> str:
    i2t = metrics["I2T"]
    t2i = metrics["T2I"]
    return (
        f"{level} | I2T R@1 {i2t['R@1']:.3f} R@5 {i2t['R@5']:.3f} "
        f"R@10 {i2t['R@10']:.3f} | T2I R@1 {t2i['R@1']:.3f} "
        f"R@5 {t2i['R@5']:.3f} R@10 {t2i['R@10']:.3f} | "
        f"mR {metrics['mR']:.3f}"
    )


class Evaluator:
    """Global-feature evaluation with INSTANCE-level retrieval metrics."""

    def __init__(self, img_loader, txt_loader, output_dir=None, split_name="val"):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.output_dir = output_dir
        self.split_name = split_name
        self.logger = logging.getLogger("Herb-LEGA.eval")

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _compute_embeddings(self, model):
        model = self._unwrap(model).eval()
        device = next(model.parameters()).device

        image_features = []
        image_ids = []
        with torch.no_grad():
            for batch in self.img_loader:
                images = batch["images"].to(device, non_blocking=True)
                features = model.encode_image(images)
                image_features.append(features.float().cpu())
                image_ids.append(batch["image_ids"].view(-1).cpu())

        text_features = []
        text_image_ids = []
        with torch.no_grad():
            for batch in self.txt_loader:
                captions = batch["caption_ids"].to(device, non_blocking=True)
                features = model.encode_text(captions)
                text_features.append(features.float().cpu())
                text_image_ids.append(batch["image_ids"].view(-1).cpu())

        return (
            torch.cat(image_features, dim=0),
            torch.cat(text_features, dim=0),
            torch.cat(image_ids, dim=0).long(),
            torch.cat(text_image_ids, dim=0).long(),
        )

    def eval(self, model, epoch=None):
        (
            image_features,
            text_features,
            image_ids,
            text_image_ids,
        ) = self._compute_embeddings(model)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        similarity_t2i = text_features @ image_features.t()

        instance = _compute_level_metrics(similarity_t2i, text_image_ids, image_ids)
        results = {"INSTANCE": instance}

        self.logger.info(format_result_line("INSTANCE", instance))

        table = PrettyTable(
            ["level", "I2T R@1", "I2T R@5", "I2T R@10", "T2I R@1", "T2I R@5", "T2I R@10", "mR"]
        )
        for level, metrics in results.items():
            table.add_row(
                [
                    level,
                    f"{metrics['I2T']['R@1']:.3f}",
                    f"{metrics['I2T']['R@5']:.3f}",
                    f"{metrics['I2T']['R@10']:.3f}",
                    f"{metrics['T2I']['R@1']:.3f}",
                    f"{metrics['T2I']['R@5']:.3f}",
                    f"{metrics['T2I']['R@10']:.3f}",
                    f"{metrics['mR']:.3f}",
                ]
            )
        self.logger.info("\n" + str(table))

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            suffix = f"epoch_{epoch:03d}" if isinstance(epoch, int) else "final"
            path = os.path.join(self.output_dir, f"metrics_{self.split_name}_{suffix}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)

        return results
