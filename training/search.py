"""
Range-based hyperparameter search with comparison graphs.

Trains one model per combination of (lr, dropout, units) and plots
val_auc curves so you can visually pick the best config.

Usage:
    python -m training.search            # all SEARCH ranges
    python -m training.search --quick    # lr only, faster

Results saved to tuning/search_results.png
Best config printed at the end — copy into training/config.py.
"""
import argparse
import itertools
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from training.config import (
    DENSE_UNITS,
    DROPOUT,
    LEARNING_RATE,
    SAVED_MODEL_DIR,
    SEARCH,
)
from training.data import compute_class_weights, create_data_generators
from training.models import CNNModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Search device: %s", DEVICE)

if DEVICE.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.set_num_threads(8)
    log.info("GPU memory capped at 50%% — threads: 8")

RESULTS_DIR  = SAVED_MODEL_DIR.parent / "tuning"
RESULTS_DIR.mkdir(exist_ok=True)
SEARCH_EPOCHS = 5


def _pos_weight(class_weights: dict[int, float]) -> torch.Tensor:
    return torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(DEVICE)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
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
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, auc


def _trial(
    lr: float,
    dropout: float,
    units: int,
    filters: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: dict[int, float],
) -> tuple[float, dict[str, list[float]]]:
    model     = CNNModel(filters=filters, units=units, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(class_weights))

    history: dict[str, list[float]] = {"val_auc": [], "val_loss": []}
    best_auc      = 0.0
    patience_count = 0

    for epoch in range(1, SEARCH_EPOCHS + 1):
        _run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_auc = _run_epoch(model, val_loader, criterion)
        history["val_auc"].append(vl_auc)
        history["val_loss"].append(vl_loss)
        log.debug(
            "  epoch %d/%d — val_loss: %.4f  val_auc: %.4f",
            epoch, SEARCH_EPOCHS, vl_loss, vl_auc,
        )
        if vl_auc > best_auc:
            best_auc       = vl_auc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 3:
                log.debug("  Early stop at epoch %d", epoch)
                break

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_auc, history


def run_search(quick: bool = False) -> None:
    train_loader, val_loader, _ = create_data_generators()
    class_weights = compute_class_weights(train_loader)

    lr_range      = SEARCH["lr"]
    dropout_range = SEARCH["dropout"] if not quick else [DROPOUT]
    units_range   = SEARCH["units"]   if not quick else [DENSE_UNITS]
    filters_range = SEARCH["filters"] if not quick else [32]

    combos = list(itertools.product(lr_range, dropout_range, units_range, filters_range))
    log.info("Starting search — combos: %d  epochs_per_trial: %d", len(combos), SEARCH_EPOCHS)

    results: list[dict] = []
    for i, (lr, dropout, units, filters) in enumerate(combos, 1):
        label = f"lr={lr:.0e} drop={dropout} units={units} f={filters}"
        log.info("[%02d/%02d] %s", i, len(combos), label)

        best_auc, history = _trial(lr, dropout, units, filters,
                                   train_loader, val_loader, class_weights)
        log.info("         best val_auc: %.4f", best_auc)

        results.append({
            "label": label, "lr": lr, "dropout": dropout,
            "units": units, "filters": filters,
            "best_auc": best_auc, "history": history,
        })

    _plot(results)
    _print_best(results)


def _plot(results: list[dict]) -> None:
    n    = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat: list = [axes] if n == 1 else list(np.array(axes).flatten())

    for ax, r in zip(axes_flat, results):
        ax.plot(r["history"]["val_auc"],  label="val auc",  linewidth=2)
        ax.plot(r["history"]["val_loss"], label="val loss", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(f'{r["label"]}\nbest={r["best_auc"]:.4f}', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlabel("epoch")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Hyperparameter Search — val AUC & loss per config", fontsize=12)
    plt.tight_layout()
    path = RESULTS_DIR / "search_results.png"
    plt.savefig(path, dpi=120)
    log.info("Graph saved — path: %s", path)
    plt.show()


def _print_best(results: list[dict]) -> None:
    best = max(results, key=lambda r: r["best_auc"])
    log.info(
        "Best config — val_auc: %.4f  lr: %g  dropout: %.2f  units: %d  filters: %d",
        best["best_auc"], best["lr"], best["dropout"], best["units"], best["filters"],
    )
    print("\n══ Best config ══")
    print(f"  val_auc  : {best['best_auc']:.4f}")
    print(f"  lr       : {best['lr']}")
    print(f"  dropout  : {best['dropout']}")
    print(f"  units    : {best['units']}")
    print(f"  filters  : {best['filters']}")
    print("\nUpdate training/config.py with these values, then run: make train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="only sweep lr, keep other params fixed")
    args = parser.parse_args()
    run_search(quick=args.quick)
