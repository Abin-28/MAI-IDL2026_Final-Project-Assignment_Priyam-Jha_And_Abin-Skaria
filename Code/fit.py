"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import sys  # FEATURE 8
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR  # FEATURE 9

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        
        # BUGFIX 14: The loop counter was named 'sum', shadowing Python's built-in sum()
        # function in the local scope — shadowed variables are an explicit silent bug
        # category for this assignment.

        # Change: Renamed 'sum' to 'total' throughout train_one_epoch, consistent with
        # the variable name already used correctly in evaluate().

        # Effect: Removes the shadowed built-in and makes train_one_epoch consistent
        # with evaluate() in naming conventions, improving readability and safety.
        
        correct, total = 0, 0  # BUGFIX 14
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # BUGFIX 13: Previously, optimizer.zero_grad() was missing before the forward pass,
            # causing gradients to accumulate across all batches instead of being reset each
            # iteration, leading to incorrect parameter updates and erratic training behaviour.

            # Change: Added self.optimizer.zero_grad() at the start of each training iteration,
            # before the forward pass, so each batch computes clean, independent gradients.

            # Effect: Each parameter update is now based only on the current batch's gradients,
            # preventing gradient explosion and allowing all three models to train correctly.

            self.optimizer.zero_grad()  # BUGFIX 13
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)  # BUGFIX 14
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, (correct / total) * 100  # BUGFIX 14

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs, checkpoint_path=None):  # FEATURE 8
        print("\n Starting Training Routine...", file=sys.stderr)  # FEATURE 8
        print("-" * 50, file=sys.stderr)  # FEATURE 8

        # FEATURE 8: Previously, fit() had no checkpoint saving, so the best-epoch
        # weights were never persisted to disk and were lost after training completed.

        # Change: Added an optional checkpoint_path=None argument to fit(), tracked
        # the best validation loss across epochs, and saved self.model.state_dict()
        # to checkpoint_path whenever a new best is found.

        # Effect: The best-performing model weights are written to disk during training,
        # making them available for transfer learning and later evaluation without
        # retraining from scratch.

        best_val_loss = float("inf")  # FEATURE 8

        # FEATURE 9: Previously, the learning rate remained fixed throughout training,
        # which can cause the optimiser to overshoot near convergence and settle at a
        # suboptimal minimum.

        # Change: Added CosineAnnealingLR scheduler initialised once in fit() with
        # T_max=epochs, and called scheduler.step() after each epoch.

        # Effect: The learning rate follows a cosine decay schedule across the full
        # training run, helping models converge more stably and reach a better minimum
        # than a constant learning rate.

        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)  # FEATURE 9
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            if val_loss < best_val_loss:  # FEATURE 8
                best_val_loss = val_loss   # FEATURE 8
                if checkpoint_path is not None: # FEATURE 8
                    torch.save(self.model.state_dict(), checkpoint_path) # FEATURE 8
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%",
                  file=sys.stderr, flush=True)  # FEATURE 8

            scheduler.step()  # FEATURE 9
        
        print("-" * 50, file=sys.stderr)  # FEATURE 8
        print("Training Complete!", file=sys.stderr)  # FEATURE 8
        return best_val_loss  # FEATURE 8