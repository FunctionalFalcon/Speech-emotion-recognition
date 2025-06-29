import torch, torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from ravdess_loader import RAVDESSDataSet, collate_pad
from models.cnn_lstm import CRNN
from models.cnn_eclr import ECLR

from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
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
    return np.mean(losses), ua, acc, f1, all_gts, all_preds

def save_visualizations(model_name, all_gts, all_preds, train_accs, val_accs, train_losses, val_losses, class_to_idx):
    result_dir = os.path.join("artifacts", model_name, "results")
    os.makedirs(result_dir, exist_ok=True)

    idx_to_class = {v:k for k,v in class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    #Confusion Matrix
    cm = confusion_matrix(all_gts, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= labels)
    fig, ax = plt.subplots(figsize = (10, 8))
    disp.plot(ax = ax, cmap="Blues", xticks_rotation = 45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent  
TRAIN_DIR = REPO_ROOT / "augmented_data/RAVDESS/train"
VAL_DIR = REPO_ROOT / "augmented_data/RAVDESS/val"
BATCH_SIZE = 128
LR = 0.0001
EPOCHS = 50

#  Main
# ---------------------------
if __name__ == "__main__":
    train_dataset = RAVDESSDataSet(dir = TRAIN_DIR, features="not mel")
    val_dataset = RAVDESSDataSet(dir = VAL_DIR, features="not mel")  
    train_dl = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    val_dl   = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    model = ECLR(n_classes = 8).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()

    # Display total params 
    num_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {num_params:,}")
    print("-" * 50)


    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    best=0
    best_state = None
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_ac, tr_f1 = train_epoch(model, train_dl, crit, optim, DEVICE)
        val_loss, ua, ac, val_f1, all_gts, all_preds = evaluate(model, val_dl, crit, DEVICE)
        if ac > best:
            best = ac
            best_state = model.state_dict().copy()
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_ac)
        val_accs.append(ac)
        print(f"Epoch {epoch:02}/{EPOCHS}  train_loss {tr_loss:.3f} | train_acc {tr_ac:.3f} | train_f1 {tr_f1:.3f} | val_acc {ac:.3f} | val_f1 {val_f1:.3f} | val_UA {ua:.3f} | (best ac {best:.3f})")

    # save model
    save_model = True
    if save_model:
        model_name = model.__class__.__name__
        save_dir = os.path.join("artifacts", model_name, "checkpoints")
        os.makedirs(save_dir, exist_ok=True) 
        save_path = os.path.join(save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), save_path)

        # visualization
        save_visualizations(model_name, all_gts, all_preds, train_accs, val_accs, train_losses, val_losses, train_dataset.class_to_idx)
