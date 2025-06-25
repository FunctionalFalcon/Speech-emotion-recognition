import torch, torch.nn as nn
import numpy as np
import os
from ravdess_loader import RAVDESSDataSet, collate_pad
from models.cnn_lstm   import CRNN

from sklearn.metrics import recall_score, accuracy_score, f1_score
from torch.utils.data import DataLoader
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

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
    return np.mean(losses), ua, acc, f1, all_gts, all_preds

# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent  
TRAIN_DIR = REPO_ROOT / "augmented_data/RAVDESS/train"
VAL_DIR = REPO_ROOT / "augmented_data/RAVDESS/val"
BATCH_SIZE = 128
LR = 0.0001
EPOCHS = 100

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

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []


    best=0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_ac, tr_f1 = train_epoch(model, train_dl, crit, optim, DEVICE)
        val_loss, ua, ac, val_f1, all_gts, all_preds = evaluate(model, val_dl, crit, DEVICE)
        best = max(best, ac)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_ac)
        val_accs.append(ac)
        print(f"Epoch {epoch:02}/{EPOCHS}  train_loss {tr_loss:.3f} | train_acc {tr_ac:.3f} | train_f1 {tr_f1:.3f} | val_acc {ac:.3f} | val_UA {ua:.3f} | val_f1 {val_f1:.3f} (best ac {best:.3f})")




def save_visualizations(model_name, all_gts, all_preds, train_accs, val_accs, train_losses, val_losses):
    result_dir = os.path.join("artifacts", model_name, "results")
    os.makedirs(result_dir, exist_ok=True)

    #Confusion Matrix
    cm = confusion_matrix(all_gts, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    #UA Recall Score
    ua = recall_score(all_gts, all_preds, average="macro")
    print(f"UA Recall Score: {ua:.4f}\n")

    #Accuracy Plot
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "accuracy.png"))
    plt.close()

    # 4. Loss Plot
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "loss.png"))
    plt.close()




# save model
model_name = model.__class__.__name__

save_dir = os.path.join("artifacts", model_name, "checkpoints")
os.makedirs(save_dir, exist_ok=True) 
save_path = os.path.join(save_dir, f"{model_name}.pth")
torch.save(model.state_dict(), save_path)

print(f"Model saved to: {save_path}")




# model visualization
save_visualizations(model_name, all_gts, all_preds, train_accs, val_accs, train_losses, val_losses)
