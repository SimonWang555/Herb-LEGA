import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_dalle import build_dalle
from . import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .pooling import TopKPooling_1


class HerbLEGA(nn.Module):
    """Herb-LEGA: Local Feature Enhancement and Generative Feature Alignment.

    The CLIP dual encoder uses global image-text contrastive learning,
    Complementarity-Based LFE (C-LFE), Saliency-Based LFE (S-LFE),
    and frozen diffusion-prior Generative Feature Alignment (GFA).
    """

    def __init__(self, args, num_classes=163):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg["embed_dim"]

        self.prior = None
        if "gfa" in self.current_task:
            self.prior = build_dalle(args.prior_config, args.prior_checkpoint)

        self.logit_scale_ori = nn.Parameter(
            torch.tensor(math.log(1.0 / args.temperature), dtype=torch.float32)
        )
        self.logit_scale_gen = nn.Parameter(
            torch.tensor(math.log(1.0 / args.gen_temperature), dtype=torch.float32)
        )

        self.text_poolm = TopKPooling_1(args.pool_k, dim=1)
        self.image_poolm = TopKPooling_1(args.pool_k, dim=1)

    def _set_task(self):
        self.current_task = [name.strip() for name in self.args.loss_names.split("+")]
        allowed = {"itc", "gfa", "clfe", "slfe"}
        unknown = set(self.current_task) - allowed
        if unknown:
            raise ValueError(f"Unknown loss tasks: {sorted(unknown)}")
        print(f"Training Herb-LEGA with tasks: {self.current_task}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.prior is not None:
            self.prior.eval()
        return self

    @staticmethod
    def _eot_features(text_tokens: torch.Tensor, text_features: torch.Tensor):
        rows = torch.arange(text_features.shape[0], device=text_features.device)
        return text_features[rows, text_tokens.argmax(dim=-1)].float()

    def encode_image(self, image):
        features = self.base_model.encode_image(image)
        return features[:, 0, :].float()

    def encode_text(self, text):
        features = self.base_model.encode_text(text)
        return self._eot_features(text, features)

    def encode_image_cross(self, image):
        features = self.base_model.encode_image(image)
        return features[:, 0, :].float(), features.float()

    def encode_text_cross(self, text):
        features = self.base_model.encode_text(text)
        return self._eot_features(text, features), features.float()

    @staticmethod
    def _complementarity_based_lfe(
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Select K local tokens least similar to the global token and mean-pool."""
        k = min(int(k), int(local_features.shape[1]))
        if k <= 0:
            raise ValueError("K for C-LFE must be positive.")
        similarities = F.cosine_similarity(
            global_features.unsqueeze(1), local_features, dim=-1
        )
        indices = torch.topk(similarities, k=k, dim=1, largest=False).indices
        selected = local_features.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, local_features.shape[-1])
        )
        return selected.mean(dim=1).float()

    def forward(self, batch, epoch=None):
        images = batch["images"]
        caption_ids = batch["caption_ids"]
        image_tokens, text_tokens = self.base_model(images, caption_ids)

        image_global = image_tokens[:, 0, :].float()
        text_global = self._eot_features(caption_ids, text_tokens)
        image_local = image_tokens[:, 1:, :].float()
        text_local = text_tokens.float()

        scale_ori = self.logit_scale_ori.exp().clamp(max=100.0)
        scale_gen = self.logit_scale_gen.exp().clamp(max=100.0)
        ret = {
            "temperature_ori": scale_ori.reciprocal().detach(),
            "temperature_gen": scale_gen.reciprocal().detach(),
        }

        if "itc" in self.current_task:
            ret["loss_itc"] = objectives.compute_itc(
                image_global, text_global, scale_ori
            )

        if "clfe" in self.current_task:
            image_clfe = self._complementarity_based_lfe(
                image_global, image_local, self.args.image_k
            )
            text_clfe = self._complementarity_based_lfe(
                text_global, text_local, self.args.text_k
            )
            enhanced_image = torch.cat((image_global, image_clfe), dim=1)
            enhanced_text = torch.cat((text_global, text_clfe), dim=1)
            ret["loss_clfe"] = objectives.compute_itc(
                enhanced_image, enhanced_text, scale_ori
            )

        if "slfe" in self.current_task:
            image_slfe = self.image_poolm(image_local)
            text_slfe = self.text_poolm(text_local)
            enhanced_image = torch.cat((image_global, image_slfe), dim=1)
            enhanced_text = torch.cat((text_global, text_slfe), dim=1)
            ret["loss_slfe"] = objectives.compute_itc(
                enhanced_image, enhanced_text, scale_ori
            )

        if "gfa" in self.current_task:
            if self.prior is None:
                raise RuntimeError("GFA requested but diffusion prior was not built.")

            with torch.no_grad():
                generated = self.prior.sample(
                    caption_ids,
                    timesteps=self.args.prior_sample_timesteps,
                )

            if generated.ndim != 2:
                generated = generated.reshape(generated.shape[0], -1)

            generated = generated.float()
            target_dim = image_global.shape[-1]
            vision_projection = self.base_model.visual.proj.float()

            if generated.shape[-1] == target_dim:
                generated_visual = generated

            elif (
                generated.shape[-1] == vision_projection.shape[0]
                and vision_projection.shape[1] == target_dim
            ):
                generated_visual = generated @ vision_projection

            else:
                raise RuntimeError(
                    "Incompatible GFA feature dimensions: "
                    f"backbone={self.args.pretrain_choice}, "
                    f"generated={tuple(generated.shape)}, "
                    f"visual.proj={tuple(vision_projection.shape)}, "
                    f"image_global={tuple(image_global.shape)}, "
                    f"text_global={tuple(text_global.shape)}."
                )

            ret["loss_gfa"] = objectives.compute_gfa_loss(
                generated_visual,
                text_global,
                image_global,
                scale_gen,
            )

        return ret


def build_model(args, num_classes=163):
    model = HerbLEGA(args, num_classes)
    convert_weights(model.base_model)
    return model
