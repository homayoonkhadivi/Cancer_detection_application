import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

from training.config import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, TEST_DIR, TRAIN_DIR, VALIDATION_SPLIT

log = logging.getLogger(__name__)

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    _NORMALIZE,
])

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    _NORMALIZE,
])


def create_data_generators() -> tuple[DataLoader, DataLoader, DataLoader]:
    full_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=_TRAIN_TRANSFORM)
    n_total = len(full_ds)
    n_val   = int(n_total * VALIDATION_SPLIT)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # val split uses clean transform (no augmentation)
    val_ds.dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=_VAL_TRANSFORM)

    log.info("Dataset split — train: %d  val: %d", n_train, n_val)

    # Weighted sampler to handle class imbalance
    train_labels = np.array([full_ds.targets[i] for i in train_ds.indices])
    class_counts  = np.bincount(train_labels)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    log.info("Class counts — normal: %d  pneumonia: %d", class_counts[0], class_counts[1])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=_VAL_TRANSFORM)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    log.info("Test set — samples: %d", len(test_ds))

    return train_loader, val_loader, test_loader


def compute_class_weights(train_loader: DataLoader) -> dict[int, float]:
    full_ds = datasets.ImageFolder(str(TRAIN_DIR))
    labels  = np.array(full_ds.targets)
    n_total     = len(labels)
    n_normal    = int((labels == 0).sum())
    n_pneumonia = int((labels == 1).sum())
    weights = {
        0: n_total / (2 * n_normal),
        1: n_total / (2 * n_pneumonia),
    }
    log.info("Class weights — normal: %.4f  pneumonia: %.4f", weights[0], weights[1])
    return weights
