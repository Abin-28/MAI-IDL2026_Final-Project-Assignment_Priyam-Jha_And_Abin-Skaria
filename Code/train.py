"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
import os  # FEATURE 3
import time  # FEATURE 2
import random  # FEATURE 1
import numpy as np  # FEATURE 1

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from sklearn.metrics import f1_score, precision_score, recall_score  # FEATURE 6
from fit import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)
        
    # FEATURE 1: Previously, no random seed was set, causing non-deterministic
    # results across runs with identical hyperparameters, making benchmarks
    # unreliable and results non-reproducible.

    # Change: Seed PyTorch CPU, CUDA, Python random, and NumPy RNGs from
    # config["SEED"] at the start of every run.

    # Effect: All runs with the same config and seed produce identical results,
    # making benchmark comparisons fair and reproducible.

    torch.manual_seed(config.get("SEED", 33)) # FEATURE 1
    random.seed(config.get("SEED", 33)) # FEATURE 1
    np.random.seed(config.get("SEED", 33)) # FEATURE 1
    torch.backends.cudnn.deterministic = True # FEATURE 1
    torch.backends.cudnn.benchmark = False # FEATURE 1
        
    # BUGFIX 17: Previously, train.py assumed CHANNELS and NUM_CLASSES existed as
    # flat top-level config keys, which breaks once dataset-specific metadata is
    # grouped under the nested DATASETS structure in config.json.
    #
    # Change: Read the selected dataset entry once using config["DATA"], then pull
    # CHANNELS and NUM_CLASSES from that dataset-specific block instead of expecting
    # them at the top level.
    #
    # Effect: train.py now works correctly with a single structured config file,
    # allowing dataset-specific model settings to be selected without editing source code.

    dataset_cfg = config["DATASETS"][config["DATA"]]  # BUGFIX 17
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    # BUGFIX 18: Previously, the third return value of get_loaders() was discarded
    # with _, so the test set was never evaluated after training completed.
    #
    # Change: Replaced _ with test_loader in the unpack, then called
    # trainer.evaluate(test_loader) after fit() to report test loss and accuracy.
    #
    # Effect: The complete train → validate → test pipeline now runs end-to-end,
    # making test accuracy visible and verifiable against the assignment thresholds.

    train_loader, val_loader, test_loader = get_loaders(    # BUGFIX 18
        data=config["DATA"],
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"]
    )

    model_class = getattr(models, config["MODEL"])
    
    # BUGFIX 10: Previously, drop_rate was hardcoded to 0.99 directly in the model
    # call, causing 99% of neurons to be dropped every pass and preventing learning.
    #
    # Change: Read drop_rate from config so it is fully controllable per run,
    # consistent with how all other hyperparameters are handled.
    #
    # Effect: Dropout rate is now config-driven with a sensible value, allowing
    # the models to train and learn correctly.
    
    # BUGFIX 11: Previously, activation_str=None was passed as a kwarg but models.py
    # reads kwargs.get("activation", "ReLU") — the mismatched key meant the passed
    # value was silently ignored and activation could never be set via the model call.
    #
    # Change: Removed the incorrect activation_str=None kwarg — the activation is
    # now controlled via config["ACTIVATION"] passed through **config or explicitly.
    #
    # Effect: The kwarg key now matches what the model expects, making activation
    # fully configurable via the config file.
    
    model = model_class(
    in_channels=dataset_cfg["CHANNELS"], # BUGFIX 17
    num_classes=dataset_cfg["NUM_CLASSES"], # BUGFIX 17
    drop_rate=config["DROP_RATE"], # BUGFIX 10
    activation=config["ACTIVATION"] # BUGFIX 11
    ).to(device) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    
    # FEATURE 2: Previously, no training runtime was recorded, making it impossible
    # to quantify computational cost for the Green Initiative report.

    # Change: Record wall-clock time around the full fit() call using time.time().

    # Effect: train_time_s is available for the benchmark results table and
    # enables fair runtime comparison across all models and datasets.

    t_start = time.time()  # FEATURE 2
    
    # FEATURE 3: Previously, no checkpoint was saved, so the best-epoch weights
    # were lost after training and could not be reused for transfer learning.

    # Change: Create a checkpoints/ directory and pass a per-run checkpoint path
    # to fit() so it can persist the best validation-loss weights to disk.

    # Effect: Best model weights are saved and available for transfer learning
    # on the organs dataset without retraining from scratch.

    os.makedirs("checkpoints", exist_ok=True)  # FEATURE 3    
    checkpoint_path = f"checkpoints/{config['DATA']}_{config['MODEL']}_best.pth"  # FEATURE 3
    
    # FEATURE 4: Previously, no peak GPU memory metric existed, making it
    # impossible to quantify model memory efficiency for the Green Initiative report.

    # Change: Reset per-run peak memory stats before fit(), then read
    # max_memory_allocated() immediately after to capture each model in isolation.

    # Effect: peak_mem_mb reflects only the current model's training memory,
    # enabling fair memory comparison across all four models.

    torch.cuda.reset_peak_memory_stats(device)  # FEATURE 4

    best_val_loss = trainer.fit(
        train_loader, val_loader,
        epochs=config["EPOCHS"],
        checkpoint_path=checkpoint_path  # FEATURE 3
    )
    
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2  # FEATURE 4
    
    train_time = time.time() - t_start  # FEATURE 2
    
    trainer.evaluate(test_loader) # BUGFIX 18

    # FEATURE 5: Previously, no per-sample inference latency was measured, making
    # it impossible to compare model efficiency for the Green Initiative report.

    # Change: Time a full pass over the test set after training and divide by the
    # total number of samples to get milliseconds per sample.

    # Effect: latency_ms is available for the benchmark table and enables
    # fair inference speed comparison across all models.

    lat_start = time.time()  # FEATURE 5
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            model(images.to(device))
    latency_ms = (time.time() - lat_start) / len(test_loader.dataset) * 1000  # FEATURE 5

    # FEATURE 6: Previously, only accuracy was reported, which is insufficient for
    # the REPORT.md benchmark table that requires precision, recall, and macro F1.

    # Change: Run a second pass over the test set to collect all predictions and
    # ground-truth labels, then compute macro F1, precision, and recall with sklearn.

    # Effect: All four required metrics are available for every dataset-model
    # permutation, satisfying the consolidated benchmark report requirements.

    all_preds, all_labels = [], []  # FEATURE 6
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    test_f1 = f1_score(all_labels, all_preds, average="macro")  # FEATURE 6
    test_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)  # FEATURE 6
    test_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)  # FEATURE 6

    print(
        f"[RESULT] dataset={config['DATA']} | model={config['MODEL']} | "
        f"test_acc={test_acc/100:.4f} | test_f1={test_f1:.4f} | "
        f"val_loss={best_val_loss:.4f} | train_time={train_time:.2f}s | "
        f"latency={latency_ms:.4f} ms/sample"
    )

if __name__ == "__main__":
    main()