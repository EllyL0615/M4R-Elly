"""Linear probe (width-agnostic).

Ported verbatim: MethodProbe, set_seed, split_train_validation, make_dataloader,
compute_pos_weight, run_epoch, evaluate_epoch, train_probe, predict_logits, the
WIDTH-AGNOSTIC load_hidden_states (the 4096-asserting first copy is DISCARDED per migration).
input_dim = x.shape[1] (read width; do NOT assert 4096 — 1b=2048, 3b=3072, 8b=4096).

DISCARDED: CONFIG globals, make_args/argparse/parse_args, BASE_DIR absolutes, the 3-way
notebook orchestration `main` (rebuilt cleanly in experiments/run_stage0.py). train_probe now
takes explicit hyperparameters instead of an argparse.Namespace (clean wiring only; the
early-stopping / val-split / BCEWithLogits logic is unchanged).
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MethodProbe(nn.Module):
    def __init__(self, input_dim: int, num_methods: int = 27):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_methods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_hidden_states(path) -> np.ndarray:
    """WIDTH-AGNOSTIC (kept override). Accepts any rank-2 width; probe sets input_dim=x.shape[1]."""
    data = np.load(path)
    if data.ndim != 2:
        raise RuntimeError(f"Hidden states must be rank-2, got shape {data.shape} from {path}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"Hidden states contain NaN/Inf: {path}")
    print(f"[i] {Path(path).name}: hidden-state width = {data.shape[1]}")
    return data.astype(np.float32)


def split_train_validation(
    x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0:
        return x, y, x.copy(), y.copy()

    n = x.shape[0]
    val_size = max(1, int(round(n * val_ratio)))
    val_size = min(val_size, n - 1)

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    weight = (neg + 1e-6) / (pos + 1e-6)
    weight = np.clip(weight, 1.0, 100.0)
    return torch.from_numpy(weight.astype(np.float32))


def run_epoch(model, dataloader, criterion, optimizer, device) -> float:
    model.train()
    losses: List[float] = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else math.nan


def evaluate_epoch(model, dataloader, criterion, device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else math.nan


def _macro_auc(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Macro ROC-AUC over columns that have both classes present (NaN-safe)."""
    from sklearn.metrics import roc_auc_score

    probs = 1.0 / (1.0 + np.exp(-logits))
    aucs = []
    for k in range(y_true.shape[1]):
        yk = y_true[:, k]
        if 0 < yk.sum() < len(yk):
            aucs.append(roc_auc_score(yk, probs[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    val_ratio: float = 0.1,
    use_pos_weight: bool = False,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> Tuple[MethodProbe, Dict[str, float]]:
    """Train the linear probe with internal val split + early stopping (logic unchanged)."""
    x_subtrain, y_subtrain, x_val, y_val = split_train_validation(
        x_train, y_train, val_ratio=val_ratio, seed=seed,
    )

    train_loader = make_dataloader(x_subtrain, y_subtrain, batch_size, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, batch_size, shuffle=False)

    model = MethodProbe(input_dim=x_train.shape[1], num_methods=y_train.shape[1]).to(device)

    if use_pos_weight:
        criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(y_subtrain).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        if verbose:
            print(f"[i] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            if verbose:
                print(f"[i] Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint captured.")
    model.load_state_dict(best_state)

    val_logits = predict_logits(model, x_val, device, batch_size)
    metrics = {
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "val_macro_auc": _macro_auc(y_val, val_logits),
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "subtrain_rows": int(x_subtrain.shape[0]),
        "input_dim": int(x_train.shape[1]),
    }
    return model, metrics


def predict_logits(model, x: np.ndarray, device, batch_size: int) -> np.ndarray:
    loader = make_dataloader(x, np.zeros((x.shape[0], 1), dtype=np.float32), batch_size, shuffle=False)
    score_list: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            score_list.append(model(batch_x).detach().cpu().numpy())
    return np.concatenate(score_list, axis=0).astype(np.float32)
