"""Small deterministic encoders for testing the real TiSAM decoder offline."""

import pytest
import torch

from tisam.config import ModelConfig
from tisam.model import segmentor
from tisam.model.feature_pyramid import FeaturePyramidSpec


class SmallSAM(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 16, 1)
        self.finetune = kwargs.get("finetune", False)
        for parameter in self.parameters():
            parameter.requires_grad_(self.finetune)

    def forward(self, images):
        features = [
            torch.nn.functional.adaptive_avg_pool2d(self.stem(images), size)
            for size in (16, 8, 4, 2)
        ]
        return features[0], features, None, None

    def get_finetuned_params(self):
        return [p for p in self.parameters() if p.requires_grad]


class SmallExtra(torch.nn.Module):
    def __init__(self, *, extra_embed_dim, finetune=False, feature_mode="final", **kwargs):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, extra_embed_dim, 1)
        self.finetune = finetune
        self.feature_mode = feature_mode

    def configure_freezing(self):
        for parameter in self.parameters():
            parameter.requires_grad_(self.finetune)

    def forward(self, images):
        features = self.stem(torch.nn.functional.adaptive_avg_pool2d(images, 8))
        return (features, features, features) if self.feature_mode == "deep" else features

    def get_finetuned_params(self):
        return [p for p in self.parameters() if p.requires_grad]


@pytest.fixture
def small_model(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(segmentor, "build_sam3_encoder", lambda **kw: SmallSAM(**kw))
    monkeypatch.setattr(segmentor, "ExtraEncoder", SmallExtra)
    monkeypatch.setattr(
        segmentor,
        "SAM3_FEATURE_PYRAMID_SPEC",
        FeaturePyramidSpec((16, 16, 16, 16), ((16, 16), (8, 8), (4, 4), (2, 2))),
    )

    def factory(**overrides):
        values = dict(
            num_classes=2,
            extra_encoder="hf-hub:MahmoodLab/UNI2-h",
            d_model=16,
            n_heads=4,
            num_layers=2,
            dim_feedforward=32,
            output_hw=(16, 16),
            input_hw=(16, 16),
            dropout=0,
            sam_proj=True,
        )
        values.update(overrides)
        torch.manual_seed(42)
        return segmentor.TiSAM(ModelConfig(**values))

    return factory
