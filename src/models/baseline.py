import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import matplotlib.pyplot as plt

DATA_DIR = "."

RESULTS_DIR = os.path.join(DATA_DIR, "results") 

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_baseline(input_file, output_dir_name):
    print(f"\n--- Training Baseline: {output_dir_name} ---")
    INPUT_PATH = os.path.join(DATA_DIR, "data/processed", input_file)
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "baseline", output_dir_name) 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {INPUT_PATH}...")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values(['atc_l4', 'month']) 
    
    print(f"Total samples: {len(df)}")
    
    drop_cols = ['atc_l4', 'month', 'shortage', 'onset_next', 'shortage_next', 'atc_l3']
    
    existing_drop = [c for c in drop_cols if c in df.columns]
    
    feature_cols = [c for c in df.columns if c not in existing_drop]
    print(f"Features: {feature_cols}")
    
    target = 'onset_next'
    
    X = df[feature_cols]
    y = df[target]
    
    months = df['month'].unique()
    split_idx = int(len(months) * 0.8)
    split_date = months[split_idx]
    
    print(f"Splitting data at {split_date}")
    
    train_mask = df['month'] < split_date
    test_mask = df['month'] >= split_date
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Train Positive Rate: {y_train.mean():.4f}")
    print(f"Test Positive Rate: {y_test.mean():.4f}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    from src.evaluation.metrics import evaluate_model, bootstrap_auc
    
    roc_auc, pr_auc, recall_at_k, precision_at_k = evaluate_model(y_test, y_pred_prob)
    
    boot_mean, boot_std = bootstrap_auc(y_test.values, y_pred_prob, n=1000)
    
    print(f"\n--- Results ---")
    print(f"ROC-AUC: {roc_auc:.4f} (Bootstrap: {boot_mean:.4f} +/- {boot_std:.4f})")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Recall@Top10%: {recall_at_k:.4f}")
    print(f"Precision@Top10%: {precision_at_k:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"ROC-AUC-Boot-Mean: {boot_mean:.4f}\n")
        f.write(f"ROC-AUC-Boot-Std: {boot_std:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Recall@Top10%: {recall_at_k:.4f}\n")
        f.write(f"Precision@Top10%: {precision_at_k:.4f}\n")
        
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'LR (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png")) 
    plt.close() 
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coef': model.coef_[0]
    }).sort_values('coef', key=abs, ascending=False)
    print("\nFeature Importance:\n", importance)
    importance.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

if __name__ == "__main__":
    train_baseline("baseline_no_graph.csv", "no_neighbor")
    train_baseline("baseline_with_neighbor.csv", "with_neighbor")