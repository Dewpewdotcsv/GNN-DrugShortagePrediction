# Hierarchical GNN for Drug Shortage Prediction

An end-to-end Machine Learning pipeline utilizing Graph Neural Networks (GNNs) to predict early onset drug shortages. This project compares traditional statistical baselines (Logistic Regression with lag/neighbor features) against hierarchical, relation-weighted GNN architectures built on the World Health Organization's Anatomical Therapeutic Chemical (ATC) classification.

---

## 🚀 Key Results & Highlights
* **~13% Absolute AUC Improvement:** The proposed **Hierarchical Weighted GNN** achieves a **0.884 ROC-AUC**, significantly outperforming the statistical baseline (**0.781 ROC-AUC**).
* **High-Yield Early Warning:** Reaches a **52.2% Recall @ Top 10% Risk** (a 31% relative improvement over baseline models), making it a viable tool for active supply-chain surveillance.
* **Ablation Studies:** Validates that representing fine-grained drug ingredients (L5 level) and their relationships to broader therapeutic groups (L4 level) is critical—collapsing the hierarchy to a flat graph causes performance to drop to 0.667 ROC-AUC.

---

## 📁 Repository Structure
```bash
.
├── src/
│   ├── data/
│   │   ├── preprocessing.py          # Data cleaning & L4 therapeutic group mapping
│   │   ├── feature_engineering.py    # Lag, rolling mean, and statistical features
│   │   └── weighted_graph_builder.py # Builds hierarchical graphs with co-shortage correlations
│   ├── models/
│   │   ├── baseline.py               # Logistic Regression baseline with neighbor features
│   │   ├── weighted_gnn.py           # PyTorch Geometric GNN architecture definition
│   │   └── train_weighted_gnn.py     # Training and evaluation loop for the GNN
│   └── evaluation/
│       ├── metrics.py                # AUC, Recall@K, and bootstrapping validation
│       ├── ablation_study.py         # Evaluates effects of weights and graph hierarchy
│       └── visualize_embeddings.py   # UMAP dimensionality reduction of learned embeddings
├── results/                          # Evaluation plots, csv metrics, and research report
└── requirements.txt                  # Python dependencies
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-name>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔄 Reproduction Pipeline

Execute the pipeline step-by-step to preprocess data, construct graphs, train models, and run evaluation:

```bash
# 1. Clean data and map shortages to therapeutic groups
python3 src/data/preprocessing.py

# 2. Extract lag and time-series features
python3 src/data/feature_engineering.py

# 3. Fit baseline Logistic Regression model
python3 src/models/baseline.py

# 4. Construct the hierarchical weighted drug graph
python3 src/data/graph_builder.py

# 5. Train and evaluate the GNN models
python3 src/models/train_gnn.py

# 6. Run model comparisons and ablation studies
python3 src/evaluation/compare_models.py
```

---

## 📊 Performance Comparison

| Model | ROC-AUC | PR-AUC | Recall @ Top 10% | Precision @ Top 10% |
| :--- | :---: | :---: | :---: | :---: |
| **Hierarchical Weighted GNN (Ours)** | **0.8838** | **0.1191** | **52.15%** | **11.10%** |
| Logistic Regression (+ Neighbor features) | 0.7827 | 0.0785 | 37.37% | 7.00% |
| Logistic Regression (Lag features only) | 0.7805 | 0.0801 | 39.78% | 7.00% |
| Flat Weighted GCN (No Hierarchy) | 0.6665 | 0.0485 | 27.42% | 5.80% |

*Bootstrap evaluation (N=1000) confirms statistical significance ($0.8836 \pm 0.0059$ vs $0.7805 \pm 0.0050$).*

---

## 🧪 Key Insights
1. **The Power of Hierarchy:** Flat graph representations (excluding L5 ingredient nodes) underperform compared to simple logistic regression. Adding hierarchical structure allows the GNN to learn representation pathways from molecular drug ingredients to broad therapeutic classes.
2. **Co-Shortage Edges:** Incorporating weighted edges based on co-occurrence correlation prevents data sparsity and routes shortage warning signals to vulnerable neighbor classes.
