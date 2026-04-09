import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import the specific ML classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore') 

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DATA_FILE = '/home/yc47480/brainvote/dataset/eeg2vote_withinsub.npz'
RESULTS_FILE = 'results_eeg_ML_REAL_DATA.csv'

print(f"--- Brainvote: FAST EEG REAL DATA EVALUATION (Traditional ML) ---")

# Load the dictionary-style NPZ file
data = np.load(DATA_FILE)

# Get the list of subjects from the file
# Assuming your subject IDs are stored in an array inside the NPZ named 'subject_ids'
unique_subjects = np.unique(data['subject_ids'])

# Initialize the results CSV
pd.DataFrame(columns=["Subject", "Model", "Acc", "F1", "AUC"]).to_csv(RESULTS_FILE, index=False)

# Define the models (Swapped out standard Gradient Boosting for HistGradientBoosting)
models_to_test = {
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "SVM": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "Random_Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient_Boosting": HistGradientBoostingClassifier(random_state=42) # 50x faster than standard GradientBoosting
}

# ==========================================
# 2. EVALUATION LOOP
# ==========================================
def evaluate_ml_model(X, y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        classes, counts = np.unique(y_train, return_counts=True)
        target = len(y_train) // 2 
        
        # PCA dynamically compresses features to those that explain 95% of the brainwave variance,
        # making SMOTE and training significantly faster.
        try:
            under = RandomUnderSampler(sampling_strategy={c: target for c, n in zip(classes, counts) if n > target}, random_state=42)
            over = SMOTE(sampling_strategy={c: target for c, n in zip(classes, counts) if n < target}, random_state=42)
            
            pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()), 
                ('pca', PCA(n_components=0.95, random_state=42)), 
                ('under', under), 
                ('over', over), 
                ('classifier', model)
            ])
        except Exception:
            # Fallback if SMOTE fails due to extreme class imbalance in a specific fold
            pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()), 
                ('pca', PCA(n_components=0.95, random_state=42)), 
                ('classifier', model)
            ])
            
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        # Calculate probabilities for AUC
        try:
            # For binary classification, we want the probability of the positive class (column 1)
            preds_proba = pipeline.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, preds_proba))
        except (AttributeError, IndexError):
            # Fallback if model doesn't natively support predict_proba in this pipeline config
            try:
                 preds_proba = pipeline.decision_function(X_test)
                 aucs.append(roc_auc_score(y_test, preds_proba))
            except (AttributeError, IndexError):
                 aucs.append(0.5) 
        
        accs.append(accuracy_score(y_test, preds) * 100)
        f1s.append(f1_score(y_test, preds, average='macro'))
        
    return np.mean(accs), np.mean(f1s), np.mean(aucs)

# ==========================================
# 3. EXECUTION (WITH REAL LABELS)
# ==========================================
print("Starting Real Data Baseline Evaluation...\n" + "="*70)
grand_results = {model_name: {'acc': [], 'f1': [], 'auc': []} for model_name in models_to_test.keys()}

for sub_id in unique_subjects:
    
    # 1. Dynamically pull the exact data for this specific subject
    try:
        X_sub_3d = data[f"{sub_id}_X"]
        y_sub = data[f"{sub_id}_y"]
    except KeyError:
        print(f"Skipping {sub_id}: Data keys not found in NPZ.")
        continue
    
    # 2. FLATTEN THE EEG DATA
    # Transforms (Trials, Channels, Time) into (Trials, Features)
    X_sub_flat = X_sub_3d.reshape(X_sub_3d.shape[0], -1)
    
    # 3. 🚨 REAL DATA: DO NOT SHUFFLE LABELS 🚨
    # We are using the actual ground-truth labels (y_sub) directly.
    classes, counts = np.unique(y_sub, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 6: 
        print(f"Skipping {sub_id} due to lack of class variance.")
        continue
        
    print(f"\n[{sub_id}] Training on REAL labels (Raw Features: {X_sub_flat.shape[1]})...")
    
    for model_name, model in models_to_test.items():
        try:
            acc, f1, auc = evaluate_ml_model(X_sub_flat, y_sub, model)
            grand_results[model_name]['acc'].append(acc)
            grand_results[model_name]['f1'].append(f1)
            grand_results[model_name]['auc'].append(auc)
            
            print(f"  -> {model_name:<25} | Real F1: {f1:.4f} | Acc: {acc:.2f}% | AUC: {auc:.4f}")
            
            pd.DataFrame([{
                "Subject": sub_id, 
                "Model": model_name, 
                "Acc": round(acc, 2), 
                "F1": round(f1, 4),
                "AUC": round(auc, 4)
            }]).to_csv(RESULTS_FILE, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"  -> {model_name:<25} | FAILED: {str(e)}")

print("\n" + "="*70)
print("🎯 REAL DATA GRAND AVERAGE RESULTS 🎯")
print("="*70)
for model_name, metrics in grand_results.items():
    if metrics['f1']:
        avg_acc = np.mean(metrics['acc'])
        avg_f1 = np.mean(metrics['f1'])
        avg_auc = np.mean(metrics['auc'])
        print(f"{model_name:<25} | Macro F1: {avg_f1:.4f} | Acc: {avg_acc:.2f}% | AUC: {avg_auc:.4f}")