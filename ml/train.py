import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes=2, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob = torch.log_softmax(pred, dim=-1)
        return -(one_hot * log_prob).sum(dim=-1).mean()


def evaluate(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)
            logits = model(ids, mask)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    return acc, f1, prec, rec, all_preds, all_labels


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=15,
    lr=3e-4,
    patience=5,
    seed=42,
    verbose=True,
    device="cpu",
    mlflow_run=True,
    run_name=None,
    extra_params=None,
):
    set_seed(seed)
    model = model.to(device)
    criterion = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_val_acc = 0.0
    best_state = None
    patience_ctr = 0
    history = []

    params = dict(
        num_epochs=num_epochs, lr=lr, patience=patience, seed=seed, device=str(device)
    )
    if extra_params:
        params.update(extra_params)

    ctx = mlflow.start_run(run_name=run_name) if mlflow_run else _nullctx()
    with ctx:
        if mlflow_run:
            mlflow.log_params(params)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["label"].to(device)
                optimizer.zero_grad()
                logits = model(ids, mask)
                loss = criterion(logits, lbls)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, device)
            history.append({"epoch": epoch + 1, "loss": total_loss, "val_acc": val_acc})

            if mlflow_run:
                mlflow.log_metrics(
                    {"train_loss": total_loss, "val_acc": val_acc, "val_f1": val_f1},
                    step=epoch,
                )

            if verbose and (epoch % 3 == 0 or epoch == num_epochs - 1):
                print(
                    f"  Epoch {epoch+1:3d}/{num_epochs} "
                    f"| Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch+1}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        test_acc, test_f1, test_prec, test_rec, preds, labels = evaluate(
            model, test_loader, device
        )

        if mlflow_run:
            mlflow.log_metrics(
                {
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "test_precision": test_prec,
                    "test_recall": test_rec,
                    "best_val_acc": best_val_acc,
                }
            )

    return {
        "test_acc": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "test_prec": round(test_prec, 4),
        "test_rec": round(test_rec, 4),
        "val_acc": round(best_val_acc, 4),
        "preds": preds,
        "labels": labels,
        "history": history,
    }


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass
