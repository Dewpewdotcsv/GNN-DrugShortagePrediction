import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.weighted_gnn import WeightedHierarchicalGNN
from src.evaluation.metrics import evaluate_model, bootstrap_auc

DATA_DIR = "."
GRAPH_FILE = os.path.join(DATA_DIR, "data/processed/graphs/weighted_hetero_graphs.pt")
RESULTS_DIR = os.path.join(DATA_DIR, "results/weighted_gnn")

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_weighted_gnn(model_type='GCN', use_weights=True, use_hierarchy=True, use_coshortage=True, epochs=50, hidden=64, lr=0.001):
    
    run_name = f"{model_type}_weights_{use_weights}_hier_{use_hierarchy}_coshort_{use_coshortage}"
    print(f"\n--- Training {run_name} ---")
    
    if not os.path.exists(GRAPH_FILE):
        print(f"Error: {GRAPH_FILE} not found.")
        return

    graphs = torch.load(GRAPH_FILE, weights_only=False)
    print(f"Loaded {len(graphs)} snapshots.")
    
    split_idx = int(len(graphs) * 0.8)
    train_graphs = graphs[:split_idx]
    test_graphs = graphs[split_idx:]
    
    sample = graphs[0]
    num_l4 = sample['l4'].x.size(1)
    num_l5 = sample['l5'].x.size(1)
    
    model = WeightedHierarchicalGNN(num_l4, num_l5, hidden, model_type=model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    all_y = torch.cat([g['l4'].y for g in train_graphs])
    n_pos = (all_y == 1).sum().item()
    n_neg = (all_y == 0).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    print(f"Pos Weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in train_graphs:
            optimizer.zero_grad()
            
            edge_weights = {}
            if use_weights:
                if ('l4', 'correlated', 'l4') in data.edge_attr_dict:
                     edge_weights[('l4', 'correlated', 'l4')] = data['l4', 'correlated', 'l4'].edge_attr
            
            edge_index_dict = data.edge_index_dict.copy()
            
            if not use_hierarchy:
                keys_to_remove = [k for k in edge_index_dict.keys() if 'l5' in k[0] or 'l5' in k[2]]
                for k in keys_to_remove:
                    del edge_index_dict[k]
            
            if not use_coshortage:
                 keys_to_remove = [k for k in edge_index_dict.keys() if 'correlated' in k[1]]
                 for k in keys_to_remove:
                     del edge_index_dict[k]
                     
            out = model(data.x_dict, edge_index_dict, edge_weight_dict=edge_weights)
            
            loss = criterion(out.squeeze(), data['l4'].y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for data in test_graphs:
                
                edge_weights = {}
                if use_weights:
                     if ('l4', 'correlated', 'l4') in data.edge_attr_dict:
                         edge_weights[('l4', 'correlated', 'l4')] = data['l4', 'correlated', 'l4'].edge_attr
                
                edge_index_dict = data.edge_index_dict.copy()
                if not use_hierarchy:
                    keys_to_remove = [k for k in edge_index_dict.keys() if 'l5' in k[0] or 'l5' in k[2]]
                    for k in keys_to_remove:
                        del edge_index_dict[k]
                
                if not use_coshortage:
                     keys_to_remove = [k for k in edge_index_dict.keys() if 'correlated' in k[1]]
                     for k in keys_to_remove:
                         del edge_index_dict[k]
                
                out = model(data.x_dict, edge_index_dict, edge_weight_dict=edge_weights)
                all_preds.append(torch.sigmoid(out).squeeze())
                all_true.append(data['l4'].y)
                
        all_preds = torch.cat(all_preds).numpy()
        all_true = torch.cat(all_true).numpy()
        
        roc_auc, pr_auc, recall_at_k, precision_at_k = evaluate_model(all_true, all_preds)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_graphs):.4f} | AUC: {roc_auc:.4f} | PR: {pr_auc:.4f}")
            
    start_boot = True 
    if start_boot:
        boot_mean, boot_std = bootstrap_auc(all_true, all_preds, n=1000)
    else:
        boot_mean, boot_std = 0.0, 0.0

    print(f"Final Test AUC: {roc_auc:.4f} (Boot: {boot_mean:.4f} +/- {boot_std:.4f})")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Recall@Top10%: {recall_at_k:.4f}")
    print(f"Precision@Top10%: {precision_at_k:.4f}")
    
    outfile = os.path.join(RESULTS_DIR, f"{run_name}.txt")
    with open(outfile, "w") as f:
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"ROC-AUC-Boot-Mean: {boot_mean:.4f}\n")
        f.write(f"ROC-AUC-Boot-Std: {boot_std:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Recall@Top10%: {recall_at_k:.4f}\n")
        f.write(f"Precision@Top10%: {precision_at_k:.4f}\n")
        
    np.savez(os.path.join(RESULTS_DIR, f"{run_name}_preds.npz"), preds=all_preds, true=all_true)
    
    print("Extracting embeddings from last test snapshot...")
    model.eval()
    if len(test_graphs) > 0:
        last_data = test_graphs[-1]
        
        edge_weights = {}
        if use_weights:
            if ('l4', 'correlated', 'l4') in last_data.edge_attr_dict:
                 edge_weights[('l4', 'correlated', 'l4')] = last_data['l4', 'correlated', 'l4'].edge_attr
        
        edge_index_dict = last_data.edge_index_dict.copy()
        if not use_hierarchy:
            keys_to_remove = [k for k in edge_index_dict.keys() if 'l5' in k[0] or 'l5' in k[2]]
            for k in keys_to_remove:
                del edge_index_dict[k]
        
        if not use_coshortage:
             keys_to_remove = [k for k in edge_index_dict.keys() if 'correlated' in k[1]]
             for k in keys_to_remove:
                 del edge_index_dict[k]
                 
        with torch.no_grad():
            logits, embeddings = model(last_data.x_dict, edge_index_dict, edge_weight_dict=edge_weights, return_embedding=True)
            
        np.savez(os.path.join(RESULTS_DIR, f"{run_name}_embeddings.npz"), embeddings=embeddings.numpy())
        print(f"Saved embeddings to {run_name}_embeddings.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GCN", choices=["GCN", "SAGE"])
    parser.add_argument("--no_weights", action="store_true")
    parser.add_argument("--no_hierarchy", action="store_true")
    
    parser.add_argument("--no_coshortage", action="store_true") 
    args = parser.parse_args()
    
    train_weighted_gnn(
        model_type=args.model,
        use_weights=not args.no_weights,
        use_hierarchy=not args.no_hierarchy,
        use_coshortage=not args.no_coshortage
    )