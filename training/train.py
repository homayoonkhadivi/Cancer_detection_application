import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from training.config import (
    CNN_MODEL_PATH,
    EPOCHS,
    FINE_TUNE_AT,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    RESNET_MODEL_PATH,
    SAVED_MODEL_DIR,
)
from training.data import compute_class_weights, create_data_generators
from training.models import CNNModel, ResNetModel, build_cnn_model, build_resnet_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

if DEVICE.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.set_num_threads(8)
    log.info("GPU memory capped at 50%% — threads: 8")

SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _pos_weight(class_weights: dict[int, float]) -> torch.Tensor:
    return torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(DEVICE)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss: float = 0.0
    all_labels: list[float] = []
    all_probs:  list[float] = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(images)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc      = float(np.mean((np.array(all_probs) > 0.5) == np.array(all_labels)))
    return avg_loss, auc, acc


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    class_weights: dict[int, float],
    save_path: str | Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    patience: int = 6,
) -> dict[str, list[float]]:
    criterion  = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(class_weights))
    best_auc   = 0.0
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_auc, tr_acc = _run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_auc, vl_acc = _run_epoch(model, val_loader,   criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        log.info(
            "Epoch %03d/%d — train_loss: %.4f  val_loss: %.4f  train_auc: %.4f  val_auc: %.4f  train_acc: %.3f  val_acc: %.3f",
            epoch, epochs, tr_loss, vl_loss, tr_auc, vl_auc, tr_acc, vl_acc,
        )

        if vl_auc > best_auc:
            best_auc = vl_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            log.info("  Saved best model — val_auc: %.4f → %s", best_auc, save_path)
        else:
            patience_counter += 1
            log.debug("  No improvement — patience: %d/%d", patience_counter, patience)
            if patience_counter >= patience:
                log.info("  Early stopping triggered at epoch %d", epoch)
                break

        if scheduler:
            scheduler.step(vl_loss)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    log.info("Restored best weights from %s", save_path)
    return history


def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: dict[int, float],
) -> dict[str, list[float]]:
    model, optimizer = build_cnn_model()
    model = model.to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-7
    )
    log.info("=== Training CNN ===")
    history = train_loop(
        model, optimizer, train_loader, val_loader,
        EPOCHS, class_weights, CNN_MODEL_PATH, scheduler,
    )
    log.info("CNN training complete — model: %s", CNN_MODEL_PATH)
    return history


def train_resnet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: dict[int, float],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    model, optimizer = build_resnet_model()
    model = model.to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-7
    )

    log.info("=== ResNet Phase 1: frozen base ===")
    h1 = train_loop(
        model, optimizer, train_loader, val_loader,
        EPOCHS, class_weights, RESNET_MODEL_PATH, scheduler,
    )

    log.info("=== ResNet Phase 2: fine-tuning top layers ===")
    model.unfreeze_top(FINE_TUNE_AT)
    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=FINE_TUNE_LR
    )
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, factor=0.5, patience=3, min_lr=1e-7
    )
    h2 = train_loop(
        model, optimizer2, train_loader, val_loader,
        FINE_TUNE_EPOCHS, class_weights, RESNET_MODEL_PATH, scheduler2,
    )

    log.info("ResNet training complete — model: %s", RESNET_MODEL_PATH)
    return h1, h2


def plot_history(*histories: dict[str, list[float]], title: str = "") -> None:
    combined: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []
    }
    for h in histories:
        for k in combined:
            combined[k].extend(h.get(k, []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(combined["train_loss"], label="train")
    ax1.plot(combined["val_loss"],   label="val")
    ax1.set_title(f"{title} Loss".strip()); ax1.set_xlabel("epoch"); ax1.legend()

    ax2.plot(combined["train_auc"], label="train")
    ax2.plot(combined["val_auc"],   label="val")
    ax2.set_title(f"{title} AUC".strip()); ax2.set_xlabel("epoch"); ax2.legend()

    plt.tight_layout()
    out_path = SAVED_MODEL_DIR.parent / f"tuning/{title.lower().replace(' ', '_')}_history.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120)
    log.info("History plot saved — path: %s", out_path)
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_data_generators()
    class_weights = compute_class_weights(train_loader)

    cnn_history = train_cnn(train_loader, val_loader, class_weights)
    plot_history(cnn_history, title="CNN")

    h1, h2 = train_resnet(train_loader, val_loader, class_weights)
    plot_history(h1, h2, title="ResNet")
