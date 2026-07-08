"""
MAI/IDL SS26 - Final assignment.
MG 6/6/2026
"""
import csv
import json
import os
import random
import time

import numpy as np
import torch
import train
import models
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
from fit import Trainer
from sklearn.metrics import f1_score, precision_score, recall_score


CONFIG_FILE = "config.json"
RESULTS_FILE = "results.csv"
TRANSFER_RESULTS_FILE = "results_transfer.csv"


def _transfer_metrics(model, trainer, test_loader, device):
    """Run a full test-set pass and return loss, accuracy, and macro F1/precision/recall."""
    test_loss, test_acc = trainer.evaluate(test_loader)
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, y in test_loader:
            logits = model(images.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    return {
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc / 100, 4),
        "test_f1": round(f1_score(labels, preds, average="macro"), 4),
        "test_precision": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "test_recall": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
    }


def run_phase3(base_config):
    """
    Transfer-learning phase: for each architecture, train a from-scratch model on the
    organs dataset and, if a pretrained checkpoint exists, fine-tune only the classifier
    head on top of the frozen backbone. Results for both modes are appended to
    results_transfer.csv.
    """
    # Seed all RNGs so this phase is reproducible independent of the main sweep.
    seed = base_config.get("SEED", 33)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Phase-3 hyperparameters can be overridden independently of the main sweep config;
    # fall back to the base config value if no TRANSFER_* override is present.
    transfer_lr = base_config.get("TRANSFER_LEARNING_RATE", base_config["LEARNING_RATE"])
    transfer_drop_rate = base_config.get("TRANSFER_DROP_RATE", base_config["DROP_RATE"])
    transfer_batch_size = base_config.get("TRANSFER_BATCH_SIZE", base_config["BATCH_SIZE"])
    transfer_epochs = base_config.get("TRANSFER_EPOCHS", base_config["EPOCHS"])

    transfer_dataset = base_config["TRANSFER_DATASET"]
    transfer_cfg = base_config["TRANSFER_DATASET_CFG"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(
        data=transfer_dataset,
        data_path=base_config["DATA_PATH"],
        batch_size=transfer_batch_size
    )

    rows = []
    for model_name in ["AlexNet", "VGG16", "ResNet18", "GreenNet"]:
        # Baseline: train this architecture from scratch on the organs dataset
        scratch_model = getattr(models, model_name)(
            in_channels=transfer_cfg["CHANNELS"],
            num_classes=transfer_cfg["NUM_CLASSES"],
            drop_rate=transfer_drop_rate,
            activation=base_config["ACTIVATION"]
        ).to(device)

        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(scratch_model.parameters(), lr=transfer_lr)
        trainer = Trainer(scratch_model, crit, opt, device)

        ckpt = f"checkpoints/{transfer_dataset}_{model_name}_scratch_best.pth"
        t0 = time.time()
        trainer.fit(train_loader, val_loader, epochs=transfer_epochs, checkpoint_path=ckpt)
        scratch_model.load_state_dict(torch.load(ckpt, map_location=device))
        metrics = _transfer_metrics(scratch_model, trainer, test_loader, device)
        rows.append({"mode": f"{model_name}_SCRATCH", **metrics, "train_time_s": round(time.time() - t0, 2)})

        # Transfer learning: fine-tune the classifier head on a frozen, pretrained backbone
        pre_ckpt = f"checkpoints/orgs_{model_name}_best.pth"
        if os.path.exists(pre_ckpt):
            transfer_model = getattr(models, model_name)(
                in_channels=transfer_cfg["CHANNELS"],
                num_classes=transfer_cfg["NUM_CLASSES"],
                drop_rate=transfer_drop_rate,
                activation=base_config["ACTIVATION"]
            ).to(device)
            transfer_model.load_state_dict(torch.load(pre_ckpt, map_location=device))

            # Freeze the whole network, then unfreeze only the classifier head.
            for p in transfer_model.parameters():
                p.requires_grad = False
            for p in transfer_model.classifier.parameters():
                p.requires_grad = True

            opt = optim.Adam(transfer_model.classifier.parameters(), lr=transfer_lr)
            trainer = Trainer(transfer_model, crit, opt, device)

            # Keep the frozen backbone's BatchNorm layers in eval mode even when
            # model.train() is called, so their running statistics aren't disturbed
            # by the small fine-tuning dataset. Only the classifier stays in train mode.
            _base_train = transfer_model.train
            def _train_with_frozen_backbone(mode=True, _base_train=_base_train, _model=transfer_model):
                _base_train(mode)
                if mode:
                    for name, module in _model.named_children():
                        if name != "classifier":
                            module.eval()
                return _model
            transfer_model.train = _train_with_frozen_backbone

            t0 = time.time()
            ckpt = f"checkpoints/{transfer_dataset}_{model_name}_transfer_best.pth"
            trainer.fit(train_loader, val_loader, epochs=transfer_epochs, checkpoint_path=ckpt)
            transfer_model.load_state_dict(torch.load(ckpt, map_location=device))
            metrics = _transfer_metrics(transfer_model, trainer, test_loader, device)
            rows.append({"mode": f"{model_name}_TRANSFER", **metrics, "train_time_s": round(time.time() - t0, 2)})
        else:
            print(f"[Phase 3] Skipping TRANSFER for {model_name} — {pre_ckpt} not found.")
            print("          Run the full sweep first so orgs checkpoints are available.")

    # Write header once, then append this run's results.
    if not os.path.exists(TRANSFER_RESULTS_FILE):
        with open(TRANSFER_RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "test_loss", "test_acc", "test_f1", "test_precision", "test_recall", "train_time_s"])

    with open(TRANSFER_RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([
                row["mode"],
                row["test_loss"],
                row["test_acc"],
                row["test_f1"],
                row["test_precision"],
                row["test_recall"],
                row["train_time_s"],
            ])


def main():
    """
    Orchestrates the full benchmark sweep: trains and evaluates every model on every
    dataset defined in config.json, logs each run to results.csv, then runs the
    transfer-learning phase (Phase 3) on the organs dataset.
    """
    with open(CONFIG_FILE, "r") as f:
        base_config = json.load(f)

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "model", "seed",
                "test_loss", "test_acc", "test_precision", "test_recall", "test_f1",
                "val_loss", "train_time_s", "latency_ms",
                "peak_mem_train_mb", "peak_mem_inference_mb",
                "checkpoint_path",
                "returncode"
            ])

    sweep_start = time.time()
    for dataset_name in base_config["DATASETS"]:
        for model_name in base_config["MODELS"]:
            run_config = dict(base_config)
            run_config["DATA"] = dataset_name
            run_config["MODEL"] = model_name
            run_config["SEED"] = base_config.get("SEED", 33)

            print(f"\nRunning: DATA={dataset_name} | MODEL={model_name} | SEED={run_config['SEED']}")
            print("-" * 50)

            # Default/empty metrics used if the run fails, so the CSV row is still written.
            metrics = {
                "test_loss": "", "test_acc": "", "test_precision": "", "test_recall": "", "test_f1": "",
                "val_loss": "", "train_time_s": "", "latency_ms": "",
                "peak_mem_train_mb": "", "peak_mem_inference_mb": "",
                "checkpoint_path": ""
            }

            returncode = 0

            try:
                metrics = train.main(run_config)
            except Exception as e:
                print(f"[WARN] Failed: DATA={dataset_name}, MODEL={model_name} — {e}")
                returncode = 1
            finally:
                # Release GPU memory between runs so failures/large models don't
                # starve subsequent runs in the sweep.
                torch.cuda.empty_cache()

            with open(RESULTS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name,
                    model_name,
                    run_config["SEED"],
                    metrics.get("test_loss", ""),
                    metrics.get("test_acc", ""),
                    metrics.get("test_precision", ""),
                    metrics.get("test_recall", ""),
                    metrics.get("test_f1", ""),
                    metrics.get("val_loss", ""),
                    metrics.get("train_time_s", ""),
                    metrics.get("latency_ms", ""),
                    metrics.get("peak_mem_train_mb", ""),
                    metrics.get("peak_mem_inference_mb", ""),
                    metrics.get("checkpoint_path", ""),
                    returncode
                ])

    print(f"\nTotal sweep time: {time.time() - sweep_start:.2f}s")
    run_phase3(base_config)


if __name__ == "__main__":
    main()