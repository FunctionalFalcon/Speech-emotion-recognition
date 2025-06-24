import torch, torch.nn as nn
import numpy as np
from ravdess_loader import RAVDESSDataSet, collate_pad
from models.cnn_lstm   import CRNN

from sklearn.metrics import recall_score, accuracy_score, f1_score
from torch.utils.data import DataLoader
from pathlib import Path

def train_epoch(model, loader, crit, optim, device):
    model.train()
    losses = []
    preds = []
    gts = []
    for x, y, L in loader:
        x, y, L = x.to(device), y.to(device), L.to(device)
        optim.zero_grad()
        logits = model(x, L)
        loss = crit(logits, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        preds.append(logits.argmax(1).cpu())
        gts.append(y.cpu())
    
    all_preds = torch.cat(preds)
    all_gts = torch.cat(gts)
    acc = accuracy_score(all_gts, all_preds)
    f1 = f1_score(all_gts, all_preds, average='macro')
    return np.mean(losses), acc, f1

def evaluate(model, loader, crit, device):
    model.eval()  
    losses = []
    preds = []
    gts = []
    with torch.no_grad():
        for x, y, L in loader:
            x, y, L = x.to(device), y.to(device), L.to(device)
            logits = model(x, L)
            losses.append(crit(logits, y).item())
            preds.append(logits.argmax(1).cpu())
            gts.append(y.cpu())
    
    all_preds = torch.cat(preds)
    all_gts = torch.cat(gts)
    ua = recall_score(all_gts, all_preds, average="macro")
    acc = accuracy_score(all_gts, all_preds)
    f1 = f1_score(all_gts, all_preds, average='macro')
    return np.mean(losses), ua, acc, f1

# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent  
TRAIN_DIR = REPO_ROOT / "augmented_data/RAVDESS/train"
VAL_DIR = REPO_ROOT / "augmented_data/RAVDESS/val"
BATCH_SIZE = 128
LR = 0.0001
EPOCHS = 10

#  Main
# ---------------------------
if __name__ == "__main__":
    train_dl = DataLoader(RAVDESSDataSet(dir = TRAIN_DIR),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    val_dl   = DataLoader(RAVDESSDataSet(dir = VAL_DIR),
                          batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    model = CRNN(n_classes = 8).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()

    best=0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_ac, tr_f1 = train_epoch(model, train_dl, crit, optim, DEVICE)
        val_loss, ua, ac, val_f1 = evaluate(model, val_dl, crit, DEVICE)
        best = max(best, ac)
        print(f"Epoch {epoch:02}/{EPOCHS}  train_loss {tr_loss:.3f} | train_acc {tr_ac:.3f} | train_f1 {tr_f1:.3f} | val_acc {ac:.3f} | val_UA {ua:.3f} | val_f1 {val_f1:.3f} (best ac {best:.3f})")
