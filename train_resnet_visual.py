import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import mne

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# ⚠️ Update this to your mildly cleaned EEG folder
EEG_DIR = '/home/yc47480/brainvote/mildlycleaned' 
IMAGE_DIR = './stimuli'  
RESULTS_FILE = 'results_visual_PIXELS_ResNet50_WITHIN.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Brainvote: WITHIN-SUBJECT Raw Image CNN (ResNet-50) ---")
print(f"Using device: {device}\n")

# Initialize Results CSV
pd.DataFrame(columns=["Subject", "Model", "Unique_Faces", "Acc", "F1", "AUC"]).to_csv(RESULTS_FILE, index=False)

# ==========================================
# 2. CUSTOM PYTORCH IMAGE DATASET
# ==========================================
class FaceConsensusDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Standard ImageNet Normalization
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. WITHIN-SUBJECT TRAINING PIPELINE
# ==========================================
def train_within_subject_resnet(epochs=15, batch_size=32):
    
    set_files = sorted(glob.glob(os.path.join(EEG_DIR, '*.set')))
    if not set_files:
        raise FileNotFoundError(f"🛑 No .set files found in {EEG_DIR}!")

    print(f"Found {len(set_files)} subjects. Starting Within-Subject Evaluation...\n" + "="*70)
    
    grand_results = []
    
    for idx, file_path in enumerate(set_files):
        sub_id = f"Sub_{idx+1:02d}"
        
        # 1. PARSE MARKERS FOR THIS SUBJECT
        raw = mne.io.read_raw_eeglab(file_path, preload=False, verbose=False)
        sub_image_labels, sub_votes = [], []
        
        for desc in raw.annotations.description:
            if desc.startswith('S_'):
                parts = desc.split('_')
                if len(parts) >= 3:
                    try:
                        img_label = int(parts[1])
                        vote = 1 if int(parts[2]) > 0 else 0
                        sub_image_labels.append(img_label)
                        sub_votes.append(vote)
                    except ValueError:
                        continue
        
        # 2. STRICT DEDUPLICATION (Prevent Identity Leakage)
        unique_img_labels, unique_indices = np.unique(sub_image_labels, return_index=True)
        unique_votes = np.array(sub_votes)[unique_indices]
        
        # 3. VERIFY PHYSICAL FILES
        valid_image_paths = []
        valid_labels = []
        
        for img_lbl, vote in zip(unique_img_labels, unique_votes):
            img_path = os.path.join(IMAGE_DIR, f"{img_lbl}.jpg")
            if os.path.exists(img_path):
                valid_image_paths.append(img_path)
                valid_labels.append(vote)
                
        y_sub_filtered = np.array(valid_labels)
        total_unique = len(y_sub_filtered)
        
        classes, counts = np.unique(y_sub_filtered, return_counts=True)
        if len(classes) < 2 or np.min(counts) < 6:
            print(f"[{sub_id}] Skipped: Not enough class variance for 5-Fold CV.")
            continue
            
        print(f"[{sub_id}] Training on {total_unique} valid images...", end=" ", flush=True)
        
        # 4. INITIALIZE DATASET & K-FOLD
        sub_dataset = FaceConsensusDataset(valid_image_paths, y_sub_filtered, transform=data_transforms)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        indices = np.arange(len(sub_dataset))
        
        accs, f1s, aucs = [], [], []
        
        # 5. CROSS-VALIDATION LOOP
        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, y_sub_filtered)):
            train_sub = Subset(sub_dataset, train_idx)
            test_sub = Subset(sub_dataset, test_idx)
            
            # Class Weights
            y_train_fold = y_sub_filtered[train_idx]
            fold_classes, fold_counts = np.unique(y_train_fold, return_counts=True)
            
            # Safety check in case a fold is entirely one class
            if len(fold_classes) < 2:
                weights = np.array([1.0, 1.0]) 
            else:
                weights = 1.0 / fold_counts
                
            tensor_weights = torch.FloatTensor(weights / weights.sum()).to(device)
            
            # Loaders
            train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False, num_workers=2)
            
            # Model Init (Fresh model for every fold!)
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model = model.to(device)
            
            criterion = nn.CrossEntropyLoss(weight=tensor_weights)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
            
            # Train
            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs.to(device))
                    loss = criterion(outputs, labels.to(device))
                    loss.backward()
                    optimizer.step()
                    
            # Evaluate
            model.eval()
            preds, targets, probs = [], [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs.to(device))
                    _, p = torch.max(outputs, 1)
                    softmax_out = torch.softmax(outputs, dim=1)
                    
                    preds.extend(p.cpu().numpy())
                    targets.extend(labels.numpy())
                    probs.extend(softmax_out[:, 1].cpu().numpy())
                    
            fold_acc = accuracy_score(targets, preds) * 100
            fold_f1 = f1_score(targets, preds, average='macro')
            try: fold_auc = roc_auc_score(targets, probs)
            except ValueError: fold_auc = 0.50
            
            accs.append(fold_acc)
            f1s.append(fold_f1)
            aucs.append(fold_auc)
            
        # 6. LOG SUBJECT RESULTS
        mean_acc, mean_f1, mean_auc = np.mean(accs), np.mean(f1s), np.mean(aucs)
        print(f"| F1: {mean_f1:.4f} | Acc: {mean_acc:.2f}% | AUC: {mean_auc:.4f}")
        
        pd.DataFrame([{
            "Subject": sub_id, 
            "Model": "ResNet-50_Pixels", 
            "Unique_Faces": total_unique,
            "Acc": round(mean_acc, 2), 
            "F1": round(mean_f1, 4),
            "AUC": round(mean_auc, 4)
        }]).to_csv(RESULTS_FILE, mode='a', header=False, index=False)
        
        grand_results.append({"F1": mean_f1, "Acc": mean_acc, "AUC": mean_auc})

    # Print Grand Average
    if grand_results:
        df_res = pd.DataFrame(grand_results)
        print("\n" + "="*70)
        print("🎯 WITHIN-SUBJECT RAW IMAGE (RESNET-50) GRAND AVERAGE 🎯")
        print("="*70)
        print(f"Macro F1: {df_res['F1'].mean():.4f} | Acc: {df_res['Acc'].mean():.2f}% | AUC: {df_res['AUC'].mean():.4f}")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Batch size 32 is standard, drop to 16 if CUDA runs out of memory.
    # Epochs set to 12 to save a bit of compute time without sacrificing fine-tuning accuracy.
    train_within_subject_resnet(epochs=12, batch_size=32)