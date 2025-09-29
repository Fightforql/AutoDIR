from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class SAClipModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        text_ctx_len: int = 77,
        freeze_vision: bool = False,
        freeze_text: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if freeze_vision:
            for param in self.model.visual.parameters():
                param.requires_grad = False
        if freeze_text:
            for param in self.model.transformer.parameters():
                param.requires_grad = False

        self.text_ctx_len = text_ctx_len

    @torch.no_grad()
    def encode_text(self, tokenized: torch.Tensor) -> torch.Tensor:
        text_features = self.model.encode_text(tokenized)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # image_views: (batch, views, C, H, W)
        batch_size, num_views = image.shape[:2]
        flattened = image.reshape(batch_size * num_views, *image.shape[2:])
        image_features = self.model.encode_image(flattened)
        image_features = F.normalize(image_features, dim=-1)
        image_features = image_features.reshape(batch_size, num_views, -1)

        text_features = self.model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()
        return image_features, text_features, logit_scale

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        return self.tokenizer(texts)



def sa_clip_loss(
    degraded_image_features: torch.Tensor,
    clean_image_features: torch.Tensor,
    text_features: torch.Tensor,
    ground_truth_labels: torch.Tensor,
    logit_scale: torch.Tensor,
    lambda_sa: float = 1.0,
) -> torch.Tensor:

    degraded_image_features = F.normalize(degraded_image_features, dim=-1)
    clean_image_features = F.normalize(clean_image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_degraded = logit_scale * degraded_image_features @ text_features.t()
    
    logits_clean = logit_scale * clean_image_features @ text_features.t()

    p_hat_degraded = F.softmax(logits_degraded, dim=-1)
    
    p_hat_clean = F.softmax(logits_clean, dim=-1)

    fidelity_term = ground_truth_labels * p_hat_degraded
   
    loss_fid = 1 - torch.sum(torch.sqrt(fidelity_term + 1e-8), dim=-1).mean()

    semantic_agnostic_term = ground_truth_labels * p_hat_clean
    
    loss_sa = torch.sum(torch.sqrt(semantic_agnostic_term + 1e-8), dim=-1).mean()
    
    total_loss = loss_fid + lambda_sa * loss_sa
    
    return total_loss