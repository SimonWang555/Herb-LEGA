import torch
import torch.nn.functional as F


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), p=2, dim=-1)


def compute_itc(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric CLIP InfoNCE used by ITC, C-LFE and S-LFE."""
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    image_norm = _normalize(image_features)
    text_norm = _normalize(text_features)
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()
    return 0.5 * (
        F.cross_entropy(logits_per_image, labels)
        + F.cross_entropy(logits_per_text, labels)
    )


def compute_one_way_contrastive(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """One-way diagonal contrastive loss used for GFA."""
    batch_size = query_features.shape[0]
    labels = torch.arange(batch_size, device=query_features.device)
    query_norm = _normalize(query_features)
    target_norm = _normalize(target_features)
    logits = logit_scale * query_norm @ target_norm.t()
    return F.cross_entropy(logits, labels)


def compute_gfa_loss(
    generated_visual_features: torch.Tensor,
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    loss_g2t = compute_one_way_contrastive(
        generated_visual_features, text_features, logit_scale
    )
    loss_g2v = compute_one_way_contrastive(
        generated_visual_features, image_features, logit_scale
    )
    return 0.5 * (loss_g2t + loss_g2v)
