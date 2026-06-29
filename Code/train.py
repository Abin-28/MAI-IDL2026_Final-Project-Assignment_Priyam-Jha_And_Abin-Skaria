"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, _ = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

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
    in_channels=config["CHANNELS"],
    num_classes=config["NUM_CLASSES"],
    drop_rate=config["DROP_RATE"], # BUGFIX 10
    activation=config["ACTIVATION"] # BUGFIX 11
    ).to(device) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

if __name__ == "__main__":
    main()