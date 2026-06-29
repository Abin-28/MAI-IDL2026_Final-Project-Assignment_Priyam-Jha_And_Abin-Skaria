"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):

    # BUGFIX 2: Previously, the path was built with f"{data}_data.pt" which does not
    # match the actual restored dataset filenames on disk (e.g. cells.pt, chest.pt).

    # Change: Build the path using f"{data}.pt" to match the actual file names
    # of the recovered datasets stored in the data directory.

    # Effect: get_loaders() can now find and open the dataset files correctly,
    # allowing training and evaluation to run instead of failing with FileNotFoundError.

    d_path = Path(data_path) / f"{data}.pt" # BUGFIX 2
    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size
    
    # BUGFIX 1: Previously, train_data was assigned the full train_images tensor,
    # meaning the validation samples (the last val_size rows) were also part of
    # the training set — the model trained and evaluated on the same data points.

    # Change: Sliced train_data and train_labels to [:val_start] so the training
    # set only contains indices 0 → val_start, and the validation set contains
    # indices val_start → N with zero overlap between the two.

    # Effect: Eliminates data leakage between the train and validation splits.
    # Validation metrics now measure true generalisation on genuinely unseen
    # samples instead of reflecting memorised training data.

    train_data = data_dict['train_images'][:val_start]  # BUGFIX 1
    train_labels = data_dict['train_labels'][:val_start]  # BUGFIX 1

    val_data = data_dict['train_images'][val_start:]
    val_labels = data_dict['train_labels'][val_start:]
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader