| ID | Filename | Problem manifestation | Root cause (math/logic) | Correction implemented | Commit hash |
| --- | --- | --- | --- | --- | --- |
| BUGFIX_1 | data.py | Validation set reused part of the training set (data leakage), causing validation metrics to reflect training data instead of unseen data | train_data and train_labels were assigned the full tensor with no slicing, so val_data (sliced from [val_start:]) was a strict subset of train_data | Sliced train_data/train_labels to [:val_start] and kept val_data/val_labels at [val_start:], ensuring fully disjoint train and validation splits | [583268c](https://github.com/Abin-28/MAI-IDL2026_Final-Project-Assignment_Priyam-Jha_And_Abin-Skaria/commit/583268c46b166e94c9432f68e70a13a63567fae9) |

