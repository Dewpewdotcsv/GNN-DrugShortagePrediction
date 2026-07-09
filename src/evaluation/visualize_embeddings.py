import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import os

DATA_DIR = "."
L4_FILE = os.path.join(DATA_DIR, "data/processed/dispensed_l4_aggregated.csv")
EMBEDDING_FILE = os.path.join(DATA_DIR, "results/weighted_gnn/GCN_weights_True_hier_True_coshort_True_embeddings.npz")
OUTPUT_PLOT = os.path.join(DATA_DIR, "results/embedding_umap.png")

def visualize():
    print("--- Generating Embedding Visualization ---")
    
    if not os.path.exists(EMBEDDING_FILE):
        print(f"Error: {EMBEDDING_FILE} not found. Run ablation metrics first.")
        return

    data = np.load(EMBEDDING_FILE)
    embeddings = data['embeddings'] 
    print(f"Loaded embeddings: {embeddings.shape}")
    
    df = pd.read_csv(L4_FILE)
    
    all_l4 = sorted(df['atc_l4'].unique())
    
    if len(all_l4) != embeddings.shape[0]:
        print(f"Warning: Number of nodes {len(all_l4)} != embeddings {embeddings.shape[0]}")
        
    node_map = pd.DataFrame({'atc_l4': all_l4})
    
    freq = df.groupby('atc_l4')['shortage'].mean().reset_index()
    freq.rename(columns={'shortage': 'freq'}, inplace=True)
    
    node_meta = pd.merge(node_map, freq, on='atc_l4', how='left').fillna(0)
    
    colors = node_meta['freq'].values
    
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                         c=colors, cmap='viridis', 
                         s=20, alpha=0.7)
    plt.colorbar(scatter, label='Shortage Frequency')
    plt.title("GNN Embeddings (L4 Therapeutic Groups)\nColored by Shortage Risk")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    visualize()