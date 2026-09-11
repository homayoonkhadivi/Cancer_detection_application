"""
Test-set evaluation with confusion matrix and prediction visualisations.

Usage:
    python -m training.evaluate            # both CNN and ResNet
    python -m training.evaluate --model cnn
    python -m training.evaluate --model resnet

Outputs saved to tuning/
  - {model}_confusion_matrix.png
  - {model}_predictions.png   (grid: correct green / incorrect red)
  - {model}_errors.png        (false positives + false negatives)
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training.config import (
    CNN_MODEL_PATH,
    DENSE_UNITS,
    DROPOUT,
    IMG_HEIGHT,
    IMG_WIDTH,
    RESNET_MODEL_PATH,
    SAVED_MODEL_DIR,
    TEST_DIR,
)
from training.models import CNNModel, ResNetModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR   = SAVED_MODEL_DIR.parent / "tuning"
OUT_DIR.mkdir(exist_ok=True)
CLASS_NAMES = ["Normal", "Pneumonia"]

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_test_loader(batch_size: int = 32) -> tuple[DataLoader, datasets.ImageFolder]:
    ds = datasets.ImageFolder(str(TEST_DIR), transform=_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    log.info("Test set — samples: %d  classes: %s", len(ds), ds.classes)
    return loader, ds


def _load_model(name: str) -> nn.Module:
    if name == "cnn":
        model = CNNModel(units=DENSE_UNITS, dropout=DROPOUT)
        path  = CNN_MODEL_PATH
    else:
        model = ResNetModel(units=DENSE_UNITS, dropout=DROPOUT)
        path  = RESNET_MODEL_PATH

    model.load_state_dict(torch.load(str(path), map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    log.info("Loaded %s — path: %s", name.upper(), path)
    return model


def _run_inference(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_labels: list[int]   = []
    all_preds:  list[int]   = []
    all_probs:  list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    model_name: str,
) -> None:
    cm  = confusion_matrix(labels, preds)
    auc = roc_auc_score(labels, probs)
    acc = (labels == preds).mean()

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)   # recall for pneumonia
    specificity = tn / (tn + fp)   # recall for normal

    log.info(
        "%s test — auc: %.4f  acc: %.4f  sensitivity: %.4f  specificity: %.4f",
        model_name.upper(), auc, acc, sensitivity, specificity,
    )
    print("\n" + classification_report(labels, preds, target_names=CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted label", ylabel="True label",
        title=(
            f"{model_name.upper()} — Confusion Matrix\n"
            f"AUC={auc:.4f}  Acc={acc:.4f}  "
            f"Sensitivity={sensitivity:.4f}  Specificity={specificity:.4f}"
        ),
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100
            ax.text(
                j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold",
            )

    plt.tight_layout()
    out = OUT_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out, dpi=150)
    log.info("Confusion matrix saved — path: %s", out)
    plt.show()


# ── Prediction grid ───────────────────────────────────────────────────────────

def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)
    return img


def plot_predictions(
    ds: datasets.ImageFolder,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    n_cols: int = 6,
    n_rows: int = 4,
) -> None:
    n = n_cols * n_rows
    indices = np.random.choice(len(ds), size=n, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
    fig.suptitle(
        f"{model_name.upper()} — Sample Predictions  (green=correct  red=wrong)",
        fontsize=13,
    )

    for ax, idx in zip(axes.flatten(), indices):
        img, _   = ds[idx]
        true_lbl = labels[idx]
        pred_lbl = preds[idx]
        prob     = probs[idx]
        correct  = true_lbl == pred_lbl

        ax.imshow(_denormalise(img), cmap="gray")
        ax.axis("off")
        color = "#2ecc71" if correct else "#e74c3c"
        ax.set_title(
            f"True: {CLASS_NAMES[true_lbl]}\n"
            f"Pred: {CLASS_NAMES[pred_lbl]} ({prob:.2f})",
            fontsize=7,
            color=color,
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    out = OUT_DIR / f"{model_name}_predictions.png"
    plt.savefig(out, dpi=150)
    log.info("Prediction grid saved — path: %s", out)
    plt.show()


def plot_errors(
    ds: datasets.ImageFolder,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    n_each: int = 8,
) -> None:
    fp_idx = np.where((labels == 0) & (preds == 1))[0]   # Normal predicted as Pneumonia
    fn_idx = np.where((labels == 1) & (preds == 0))[0]   # Pneumonia predicted as Normal

    log.info(
        "%s errors — false_positives: %d  false_negatives: %d",
        model_name.upper(), len(fp_idx), len(fn_idx),
    )

    def _sample(idx: np.ndarray) -> np.ndarray:
        return idx if len(idx) <= n_each else np.random.choice(idx, n_each, replace=False)

    fp_sample = _sample(fp_idx)
    fn_sample = _sample(fn_idx)
    total     = len(fp_sample) + len(fn_sample)
    if total == 0:
        log.info("No errors to display for %s", model_name.upper())
        return

    cols = max(len(fp_sample), len(fn_sample), 1)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.5, 6))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    fig.suptitle(
        f"{model_name.upper()} — Errors\n"
        f"Top: False Positives (Normal→Pneumonia)  |  Bottom: False Negatives (Pneumonia→Normal)",
        fontsize=11,
    )

    for col, idx in enumerate(fp_sample):
        img, _ = ds[idx]
        axes[0, col].imshow(_denormalise(img), cmap="gray")
        axes[0, col].set_title(f"conf={probs[idx]:.2f}", fontsize=7, color="#e74c3c")
        axes[0, col].axis("off")

    for col, idx in enumerate(fn_sample):
        img, _ = ds[idx]
        axes[1, col].imshow(_denormalise(img), cmap="gray")
        axes[1, col].set_title(f"conf={probs[idx]:.2f}", fontsize=7, color="#e74c3c")
        axes[1, col].axis("off")

    for ax in axes.flatten()[len(fp_sample):cols]:
        ax.axis("off")
    for ax in axes[1].flatten()[len(fn_sample):]:
        ax.axis("off")

    plt.tight_layout()
    out = OUT_DIR / f"{model_name}_errors.png"
    plt.savefig(out, dpi=150)
    log.info("Error grid saved — path: %s", out)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(model_name: str) -> None:
    log.info("=== Evaluating %s on test set ===", model_name.upper())
    loader, ds = _load_test_loader()
    model      = _load_model(model_name)

    labels, preds, probs = _run_inference(model, loader)

    plot_confusion_matrix(labels, preds, probs, model_name)
    plot_predictions(ds, labels, preds, probs, model_name)
    plot_errors(ds, labels, preds, probs, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["cnn", "resnet", "both"], default="both",
        help="Which model to evaluate",
    )
    args = parser.parse_args()

    if args.model == "both":
        evaluate("cnn")
        evaluate("resnet")
    else:
        evaluate(args.model)
