from pathlib import Path
import torch


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict", "ema_model", "ema"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def build_dalle(config_path: str, checkpoint_path: str):
    """Build the frozen LAION DALL-E 2 diffusion prior used by Herb-LEGA.

    Expected files are ``prior_config.json`` and ``best.pth`` (or ``latest.pth``)
    downloaded from laion/DALLE2-PyTorch/prior.
    """
    try:
        from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig
    except ImportError as exc:
        raise ImportError(
            "dalle2-pytorch is required for GFA. Install requirements.txt before training/testing."
        ) from exc

    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Diffusion prior config not found: {config_file}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Diffusion prior checkpoint not found: {checkpoint_file}")

    prior_config = TrainDiffusionPriorConfig.from_json_path(str(config_file)).prior
    prior = prior_config.create()
    checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported prior checkpoint object: {type(state_dict).__name__}")
    try:
        prior.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        incompatible = prior.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if not key.startswith("clip.")]
        if unexpected or missing:
            raise RuntimeError(
                "Diffusion prior checkpoint is incompatible. "
                f"unexpected={unexpected[:10]}, missing_non_clip={missing[:10]}"
            )
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    return prior
