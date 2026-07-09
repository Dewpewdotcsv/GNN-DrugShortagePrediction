import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

DATA_DIR = "."
L4_FILE = os.path.join(DATA_DIR, "data/processed/dispensed_l4_aggregated.csv")
L5_FILE = os.path.join(DATA_DIR, "data/processed/dispensed_l5.csv")
SUPERVISED_FILE = os.path.join(DATA_DIR, "data/processed/baseline_no_graph.csv") 
OUTPUT_DIR = os.path.join(DATA_DIR, "data/processed/graphs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_weighted_graphs():
    print("Loading data...")
    
    l4_df = pd.read_csv(SUPERVISED_FILE)
    l4_df['month'] = pd.to_datetime(l4_df['month'])
    
    l5_df = pd.read_csv(L5_FILE)
    l5_df['month'] = pd.to_datetime(l5_df['month'])

    all_l4 = sorted(l4_df['atc_l4'].unique())
    l4_to_idx = {l4: i for i, l4 in enumerate(all_l4)}
    num_l4 = len(all_l4)
    
    all_l5 = sorted(l5_df['atc_l5'].unique())
    l5_to_idx = {l5: i for i, l5 in enumerate(all_l5)}
    num_l5 = len(all_l5)
    
    print(f"Nodes: {num_l4} L4, {num_l5} L5")
    
    print("Building Therapeutic Edges...")
    edge_index_l4_l4 = []
    l3_map = {}
    for l4 in all_l4:
        l3 = l4[:4]
        if l3 not in l3_map: l3_map[l3] = []
        l3_map[l3].append(l4_to_idx[l4])
        
    for l3, nodes in l3_map.items():
        if len(nodes) > 1:
            for i in nodes:
                for j in nodes:
                    if i != j:
                        edge_index_l4_l4.append([i, j])
    edge_index_l4_l4 = torch.tensor(edge_index_l4_l4, dtype=torch.long).t().contiguous()
    
    print("Building Hierarchical Edges...")
    edge_index_l5_l4 = []
    for l5 in all_l5:
        
        parent = l5[:5] 
        if parent in l4_to_idx:
            edge_index_l5_l4.append([l5_to_idx[l5], l4_to_idx[parent]])
    edge_index_l5_l4 = torch.tensor(edge_index_l5_l4, dtype=torch.long).t().contiguous()
    
    months = sorted(l4_df['month'].unique())
    split_idx = int(len(months) * 0.8)
    split_date = months[split_idx]
    print(f"Splitting at {split_date} for strict separation.")
    
    train_mask = l4_df['month'] < split_date
    l4_train = l4_df[train_mask]
    
    feat_cols_l4 = ['patients', 'rx_count', 'days_of_supply', 'dos_per_patient', 'shortage_lag1', 'shortage_roll3']
    
    feat_cols_l5 = ['patients', 'rx_count', 'days_of_supply', 'dos_per_patient']
    
    scaler_l4 = StandardScaler()
    scaler_l4.fit(l4_train[feat_cols_l4])
    l4_df[feat_cols_l4] = scaler_l4.transform(l4_df[feat_cols_l4])
    
    l5_train_mask = l5_df['month'] < split_date
    scaler_l5 = StandardScaler()
    scaler_l5.fit(l5_df.loc[l5_train_mask, feat_cols_l5])
    l5_df[feat_cols_l5] = scaler_l5.transform(l5_df[feat_cols_l5])

    print("Building Co-Shortage Weighted Edges (Train Only)...")
    pivot_df = l4_train.pivot(index='month', columns='atc_l4', values='shortage').fillna(0)
    
    corr_matrix = pivot_df.corr().values 
    
    THRESHOLD = 0.2 
    
    rows, cols = np.where(corr_matrix > THRESHOLD)
    weights = []
    co_shortage_edges = []
    
    for r, c in zip(rows, cols):
        if r != c:
            co_shortage_edges.append([r, c])
            weights.append(corr_matrix[r, c])
            
    if co_shortage_edges:
        edge_index_weighted = torch.tensor(co_shortage_edges, dtype=torch.long).t().contiguous()
        edge_attr_weighted = torch.tensor(weights, dtype=torch.float)
    else:
        edge_index_weighted = torch.empty((2, 0), dtype=torch.long)
        edge_attr_weighted = torch.empty((0), dtype=torch.float)
        
    print(f"Weighted Edges: {edge_index_weighted.size(1)} (Threshold > {THRESHOLD})")

    graph_list = []
    months = sorted(l4_df['month'].unique())
    
    for month in months:
        data = HeteroData()
        
        m_l4 = l4_df[l4_df['month'] == month]
        
        nodes_df = pd.DataFrame({'atc_l4': all_l4, 'idx': range(num_l4)})
        merged_l4 = pd.merge(nodes_df, m_l4, on='atc_l4', how='left').fillna(0)
        
        data['l4'].x = torch.tensor(merged_l4[feat_cols_l4].values, dtype=torch.float)
        data['l4'].y = torch.tensor(merged_l4['onset_next'].values, dtype=torch.long)
        
        m_l5 = l5_df[l5_df['month'] == month]
        nodes_l5 = pd.DataFrame({'atc_l5': all_l5, 'idx': range(num_l5)})
        merged_l5 = pd.merge(nodes_l5, m_l5, on='atc_l5', how='left').fillna(0)
        
        data['l5'].x = torch.tensor(merged_l5[feat_cols_l5].values, dtype=torch.float)
        
        data['l4', 'therapeutic', 'l4'].edge_index = edge_index_l4_l4
        
        data['l5', 'part_of', 'l4'].edge_index = edge_index_l5_l4
        data['l4', 'has_part', 'l5'].edge_index = edge_index_l5_l4.flip(0)
        
        data['l4', 'correlated', 'l4'].edge_index = edge_index_weighted
        data['l4', 'correlated', 'l4'].edge_attr = edge_attr_weighted
        
        data.month = str(month)
        graph_list.append(data)
        
    outfile = os.path.join(OUTPUT_DIR, "weighted_hetero_graphs.pt")
    torch.save(graph_list, outfile)
    print(f"Saved {len(graph_list)} weighted graphs to {outfile}")

if __name__ == "__main__":
    build_weighted_graphs()