import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results/weighted_gnn"
BASELINE_DIR = "results/baseline"

models = [
    
    ("GCN", True, True, "GCN_Weighted_Hierarchical"),
    ("GCN", False, True, "GCN_Unweighted_Hierarchical"),
    ("GCN", True, False, "GCN_Weighted_Flat"),
    ("SAGE", True, True, "SAGE_Weighted_Hierarchical"), 
    
]

def parse_metrics(filepath):
    metrics = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    metrics[key.strip()] = float(val)
    return metrics

def run_ablation():
    print("--- Starting Rigorous Ablation Study ---")
    
    model_c_cmd = ["python3", "src/models/train_weighted_gnn.py", "--model", "GCN", "--no_hierarchy", "--no_coshortage", "--no_weights"]
    
    model_d_cmd = ["python3", "src/models/train_weighted_gnn.py", "--model", "GCN"]

    print("Running Model C (Flat GCN)...")
    subprocess.run(model_c_cmd, check=True)
    
    print("Running Model D (Hierarchical Weighted GCN)...")
    subprocess.run(model_d_cmd, check=True)

    results = []
    
    print("Re-running Baselines (Model A & B)...")
    subprocess.run(["python3", "src/models/baseline.py"], check=True)

    path_a = os.path.join(BASELINE_DIR, "no_neighbor", "metrics.txt")
    m_a = parse_metrics(path_a)
    results.append(format_row("Logistic", m_a))
    
    path_b = os.path.join(BASELINE_DIR, "with_neighbor", "metrics.txt")
    m_b = parse_metrics(path_b)
    results.append(format_row("Logistic + Neighbor", m_b))
    
    path_c = os.path.join(RESULTS_DIR, "GCN_weights_False_hier_False_coshort_False.txt")
    m_c = parse_metrics(path_c)
    results.append(format_row("GCN", m_c))
    
    path_d = os.path.join(RESULTS_DIR, "GCN_weights_True_hier_True_coshort_True.txt")
    m_d = parse_metrics(path_d)
    results.append(format_row("Hierarchical Weighted GCN", m_d))
            
    df = pd.DataFrame(results)
    print("\n--- Final Comparison Table ---")
    print(df)
    df.to_csv("results/final_comparison.csv", index=False)
    
    if not df.empty:
        df.set_index("Model")[["ROC-AUC", "PR-AUC"]].plot(kind="barh", figsize=(10, 6))
        plt.title("Rigorous Model Comparison")
        plt.tight_layout()
        plt.savefig("results/final_comparison_plot.png")
        print("Saved plot.")

def format_row(name, m):
    
    return {
        "Model": name,
        "ROC-AUC": m.get("ROC-AUC", 0),
        "PR-AUC": m.get("PR-AUC", 0),
        "Recall@10%": m.get("Recall@Top10%", 0),
        "Precision@10%": m.get("Precision@Top10%", 0),
        "ROC-Boot-Mean": m.get("ROC-AUC-Boot-Mean", 0),
        "ROC-Boot-Std": m.get("ROC-AUC-Boot-Std", 0)
    }

if __name__ == "__main__":
    run_ablation()