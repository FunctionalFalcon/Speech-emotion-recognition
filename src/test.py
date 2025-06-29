import torch
import torch.nn as nn
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from ravdess_loader import RAVDESSDataSet, collate_pad
from models.cnn_eclr import ECLR
from models.eclra import ECLRA
from torch.utils.data import DataLoader

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Using device:", DEVICE)
if DEVICE.type == 'cuda':
    print("   > GPU:", torch.cuda.get_device_name(0))

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "artifacts/ECLRA/checkpoints/ECLRA.pth"
VAL_DIR = REPO_ROOT / "src/augmented_data/RAVDESS/val"
RESULT_DIR = REPO_ROOT / "artifacts/ECLRA/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Load dataset
val_dataset = RAVDESSDataSet(dir=VAL_DIR, features="not mel")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_pad)

# Load model
model = ECLRA(n_classes=8).to(DEVICE)
assert MODEL_PATH.exists(), f" Model file not found: {MODEL_PATH}"
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(" Model loaded successfully.")

# Evaluation
all_preds = []
all_gts = []

with torch.no_grad():
    for x, y, L in val_loader:
        x, y, L = x.to(DEVICE), y.to(DEVICE), L.to(DEVICE)
        logits = model(x, L)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_gts.append(y.cpu())

all_preds = torch.cat(all_preds)
all_gts = torch.cat(all_gts)

# Class labels
idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# Save classification report
report = classification_report(all_gts, all_preds, target_names=target_names, digits=4)
print("\n Classification Report:\n")
print(report)

report_path = RESULT_DIR / "classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Classification Report\n")
    f.write("=====================\n")
    f.write(report)

print(f" Report saved to: {report_path}")

# Save confusion matrix
cm = confusion_matrix(all_gts, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix.png")
print(f" Confusion matrix saved to: {RESULT_DIR / 'confusion_matrix.png'}")
plt.show()

"""
import torch
import torch.nn as nn
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from ravdess_loader import RAVDESSDataSet, collate_pad
from models.eclra import ECLRA  # make sure this path is correct
from torch.utils.data import DataLoader

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", DEVICE)
if DEVICE.type == 'cuda':
    print("   >", torch.cuda.get_device_name(0))

# Paths
REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "artifacts/ECLRA/checkpoints/ECLRA.pth"
VAL_DIR = REPO_ROOT / "augmented_data/RAVDESS/val"

# Load dataset
val_dataset = RAVDESSDataSet(dir=VAL_DIR, features="not mel")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_pad)

# Load model
model = ECLRA(n_classes=8).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Evaluation
all_preds = []
all_gts = []

with torch.no_grad():
    for x, y, L in val_loader:
        x, y, L = x.to(DEVICE), y.to(DEVICE), L.to(DEVICE)
        logits = model(x, L)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_gts.append(y.cpu())

# Combine batches
all_preds = torch.cat(all_preds)
all_gts = torch.cat(all_gts)

# Class labels
idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# Report
print("\n📊 Classification Report:\n")
print(classification_report(all_gts, all_preds, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(all_gts, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(REPO_ROOT / "artifacts/ECLRA/results/test_confusion_matrix.png")
plt.show()
"""