import logging
import os
from collections import OrderedDict

import torch


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=True,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _model_state_without_frozen_prior(model):
        state = model.state_dict()
        # The 1.34GB frozen prior is restored from --prior_checkpoint and is not
        # duplicated inside every Herb-LEGA training checkpoint.
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("prior.") and not key.startswith("module.prior.")
        }

    def save(self, name, **kwargs):
        if not self.save_dir or not self.save_to_disk:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        data = {"model": self._model_state_without_frozen_prior(self.model)}
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)
        save_file = os.path.join(self.save_dir, f"{name}.pth")
        self.logger.info("Saving checkpoint to %s", save_file)
        torch.save(data, save_file)

    def load(self, path=None):
        if not path:
            raise FileNotFoundError("Checkpoint path is empty.")
        self.logger.info("Loading checkpoint from %s", path)
        checkpoint = torch.load(path, map_location="cpu")
        self._load_model(checkpoint)
        return checkpoint

    def resume(self, path=None):
        checkpoint = self.load(path)
        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint

    def _load_model(self, checkpoint):
        loaded = checkpoint["model"] if "model" in checkpoint else checkpoint
        load_state_dict(self.model, loaded)


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not keys or not all(key.startswith(prefix) for key in keys):
        return state_dict
    return OrderedDict((key[len(prefix) :], value) for key, value in state_dict.items())


def load_state_dict(model, loaded_state_dict):
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, "module.")
    current = model.state_dict()
    missing_shape = []
    matched = 0
    for key, value in loaded_state_dict.items():
        candidates = [key, f"module.{key}"]
        destination = next((candidate for candidate in candidates if candidate in current), None)
        if destination is None:
            continue
        if current[destination].shape != value.shape:
            missing_shape.append((destination, tuple(value.shape), tuple(current[destination].shape)))
            continue
        current[destination] = value
        matched += 1
    if missing_shape:
        raise RuntimeError(f"Checkpoint shape mismatches: {missing_shape[:5]}")
    if matched == 0:
        raise RuntimeError("No checkpoint parameters matched the current model.")
    model.load_state_dict(current, strict=True)
