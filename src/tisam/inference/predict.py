"""Image preprocessing and dense predictions for the public inference API."""

from __future__ import annotations

import numpy as np
import torch

from tisam.model.segmentor import TiSAM


def predict(
    model: TiSAM,
    image: np.ndarray,
    *,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    probabilities: bool = False,
) -> np.ndarray:
    """Normalize one uint8 HWC RGB image and return HW labels or CHW probabilities."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    if any(v <= 0 for v in std):
        raise ValueError("std must be positive")
    device = next(model.parameters()).device
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
    tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            batch = tensor.unsqueeze(0).to(device)
            if tuple(batch.shape[-2:]) != tuple(model.input_hw):
                batch = torch.nn.functional.interpolate(
                    batch, size=model.input_hw, mode="bilinear", align_corners=False
                )
            logits = model(batch)
            logits = torch.nn.functional.interpolate(
                logits, size=image.shape[:2], mode="bilinear", align_corners=False
            )
            result = logits.softmax(1)[0] if probabilities else logits.argmax(1)[0]
            return result.float().cpu().numpy() if probabilities else result.cpu().numpy()
    finally:
        model.train(was_training)
