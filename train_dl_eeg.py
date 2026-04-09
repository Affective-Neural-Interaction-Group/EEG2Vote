import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Braindecode Models
from braindecode.models import EEGNet, ShallowFBCSPNet, EEGConformer, BIOT

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
DATA_FILE = 'eeg2vote_withinsub.npz'
RESULTS_FILE = 'results_within_subject_all_models.csv'
SFREQ = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Brainvote: Unified Within-Subject Benchmark ---")
print(f"Using device: {device}\n")

print(f"Loading dynamic channel dataset from {DATA_FILE}...")
data = np.load(DATA_FILE)
valid_subjects = data['valid_subjects']

print(f"Found {len(valid_subjects)} viable subjects.\n")

pd.DataFrame(columns=["Subject", "Model", "Trials_Per_Class", "Channels", "Acc", "F1"]).to_csv(RESULTS_FILE, index=False)

# ==========================================
# 2. DYNAMIC MODEL FACTORY
# ==========================================
def get_model(model_name, n_classes, n_chans, n_times):
    """Instantiates the requested model dynamically sizing the input channels"""
    if model_name == "EEGNet":
        return EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    elif model_name == "ShallowFBCSPNet":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    elif model_name == "EEGConformer":
        return EEGConformer(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, sfreq=SFREQ)
    elif model_name == "BIOT":
        # Note: Heavy dropout applied here specifically for within-subject scarcity
        return BIOT(n_outputs=n_classes, n_chans=n_chans, n_times=n_times, sfreq=SFREQ, 
                    hop_length=100)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

# ==========================================
# 3. UNIFIED TRAINING LOOP
# ==========================================
def train_within_subject(model_name, X, y, epochs=40, batch_size=8):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    n_chans, n_times = X.shape[1], X.shape[2]
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]).long()), 
            batch_size=batch_size, shuffle=True, drop_last=(model_name == "BIOT")
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]).long()), 
            batch_size=batch_size, shuffle=False
        )
        
        # Initialize model with the exact number of channels this subject has
        model = get_model(model_name, 2, n_chans, n_times).to(device)
        
        # Apply specific optimizer/scheduler logic based on architecture type
        if model_name == "BIOT":
            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
            total_steps = len(train_loader) * epochs
            scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.3) if total_steps > 0 else None
            criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        else:
            # Standard CNNs use standard AdamW and no label smoothing
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            scheduler = None
            criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            for b_X, b_y in train_loader:
                b_X, b_y = b_X.to(device), b_y.to(device)
                optimizer.zero_grad()
                
                out = model(b_X)
                
                # Universal Safety Catches
                if isinstance(out, tuple): out = out[0]
                if len(out.shape) == 3: out = out.squeeze(-1)
                    
                loss = criterion(out, b_y)
                loss.backward()
                
                # Gradient clipping is only strictly necessary for Transformers
                if model_name == "BIOT":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                if scheduler: scheduler.step()
                
        # Evaluation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b_X, b_y in test_loader:
                out = model(b_X.to(device))
                if isinstance(out, tuple): out = out[0]
                if len(out.shape) == 3: out = out.squeeze(-1)
                    
                _, p = torch.max(out, 1)
                preds.extend(p.cpu().numpy())
                targets.extend(b_y.numpy())
                
        # Record fold metrics
        if len(np.unique(preds)) == 1: pass 
        accs.append(accuracy_score(targets, preds) * 100)
        f1s.append(f1_score(targets, preds, average='macro'))
        
    return np.mean(accs), np.mean(f1s)

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
models_to_test = ["EEGNet", "ShallowFBCSPNet", "EEGConformer", "BIOT"]

for model_name in models_to_test:
    print("\n" + "="*60)
    print(f"EVALUATING MODEL: {model_name}")
    print("="*60)
    
    grand_accs = []
    
    for sub_id in valid_subjects:
        X_sub = data[f"{sub_id}_X"]
        y_sub = data[f"{sub_id}_y"]
        
        n_chans = X_sub.shape[1]
        trials_per_class = len(y_sub) // 2
        
        # Give CNNs 30 epochs, and Transformers 40 epochs to warm up
        target_epochs = 40 if model_name == "BIOT" else 30
        
        print(f"  Training on {sub_id} ({trials_per_class} trials/class | {n_chans} chs)...", end=" ")
        
        try:
            acc, f1 = train_within_subject(model_name, X_sub, y_sub, epochs=target_epochs, batch_size=8)
            grand_accs.append(acc)
            print(f"Acc: {acc:.2f}%")
            
            pd.DataFrame([{
                "Subject": sub_id, 
                "Model": model_name, 
                "Trials_Per_Class": trials_per_class,
                "Channels": n_chans,
                "Acc": round(acc, 2), 
                "F1": round(f1, 4)
            }]).to_csv(RESULTS_FILE, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"FAILED! Error: {str(e)}")

    if grand_accs:
        print(f"\n🎯 {model_name} Grand Average Accuracy: {np.mean(grand_accs):.2f}%\n")

print("\n✅ All models have finished within-subject evaluation!")