import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import SEMGDataset

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            trues.extend(labels.numpy())
    
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average='macro', zero_division=0)
    rec = recall_score(trues, preds, average='macro', zero_division=0)
    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    return acc, prec, rec, f1, trues, preds

def train_and_eval_model(model_fn, model_name, trial_signals, trial_labels, preprocessor, device, epochs=30):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []
    all_trues, all_preds = [], []

    print(f"\n==========================================")
    print(f"[{model_name} 5-Fold Cross Validation 시작]")
    print(f"==========================================")

    for fold, (train_idx, val_idx) in enumerate(skf.split(trial_signals, trial_labels)):
        tr_signals = [trial_signals[i] for i in train_idx]
        tr_labels = [trial_labels[i] for i in train_idx]
        val_signals = [trial_signals[i] for i in val_idx]
        val_labels = [trial_labels[i] for i in val_idx]

        train_ds = SEMGDataset(tr_signals, tr_labels, preprocessor)
        val_ds = SEMGDataset(val_signals, val_labels, preprocessor)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        model = model_fn(num_classes=5).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)

        acc, prec, rec, f1, trues, preds = evaluate(model, val_loader, device)
        print(f"Fold {fold + 1} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
        
        fold_accs.append(acc)
        fold_precs.append(prec)
        fold_recs.append(rec)
        fold_f1s.append(f1)
        all_trues.extend(trues)
        all_preds.extend(preds)

    mean_acc, mean_prec = np.mean(fold_accs), np.mean(fold_precs)
    mean_rec, mean_f1 = np.mean(fold_recs), np.mean(fold_f1s)

    print(f"\n[{model_name} 최종 평균 성능]")
    print(f"Accuracy: {mean_acc*100:.2f}% | Precision: {mean_prec*100:.2f}% | Recall: {mean_rec*100:.2f}% | F1-Score: {mean_f1*100:.2f}%")

    return {
        'model_name': model_name,
        'accuracy': mean_acc,
        'precision': mean_prec,
        'recall': mean_rec,
        'f1_score': mean_f1,
        'trues': all_trues,
        'preds': all_preds
    }