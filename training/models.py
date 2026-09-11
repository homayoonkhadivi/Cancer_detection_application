import logging

import torch
import torch.nn as nn
from torchvision import models

from training.config import CNN_FILTERS, DENSE_UNITS, DROPOUT, FINE_TUNE_AT, IMG_HEIGHT, IMG_WIDTH, LEARNING_RATE

log = logging.getLogger(__name__)


class CNNModel(nn.Module):
    def __init__(
        self,
        filters: int = CNN_FILTERS,
        units: int = DENSE_UNITS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(filters, filters * 2, 3, padding=1), nn.BatchNorm2d(filters * 2), nn.ReLU(inplace=True),
            nn.Conv2d(filters * 2, filters * 2, 3, padding=1), nn.BatchNorm2d(filters * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(filters * 2, filters * 4, 3, padding=1), nn.BatchNorm2d(filters * 4), nn.ReLU(inplace=True),
            nn.Conv2d(filters * 4, filters * 4, 3, padding=1), nn.BatchNorm2d(filters * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # infer flat size without hardcoding
        probe_size = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
        flat_size  = self.features(probe_size).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, units), nn.BatchNorm1d(units), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(units, 1),
        )
        log.debug(
            "CNNModel built — filters: %d  units: %d  dropout: %.2f  flat_size: %d",
            filters, units, dropout, flat_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNetModel(nn.Module):
    def __init__(
        self,
        units: int = DENSE_UNITS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in base.parameters():
            param.requires_grad = False

        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, units),
            nn.BatchNorm1d(units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(units, 1),
        )
        self.model = base
        log.debug(
            "ResNetModel built — units: %d  dropout: %.2f  in_features: %d",
            units, dropout, in_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def unfreeze_top(self, fine_tune_at: int = FINE_TUNE_AT) -> None:
        params = list(self.model.parameters())
        unfrozen = 0
        for param in params[fine_tune_at:]:
            param.requires_grad = True
            unfrozen += 1
        log.info("Unfrozen %d parameter groups (fine_tune_at=%d)", unfrozen, fine_tune_at)


def build_cnn_model(
    lr: float = LEARNING_RATE,
    filters: int = CNN_FILTERS,
    units: int = DENSE_UNITS,
    dropout: float = DROPOUT,
) -> tuple[CNNModel, torch.optim.Optimizer]:
    model     = CNNModel(filters=filters, units=units, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log.info("CNN model ready — lr: %g  filters: %d  units: %d  dropout: %.2f", lr, filters, units, dropout)
    return model, optimizer


def build_resnet_model(
    lr: float = LEARNING_RATE,
    units: int = DENSE_UNITS,
    dropout: float = DROPOUT,
) -> tuple[ResNetModel, torch.optim.Optimizer]:
    model     = ResNetModel(units=units, dropout=dropout)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    log.info("ResNet model ready — lr: %g  units: %d  dropout: %.2f", lr, units, dropout)
    return model, optimizer
