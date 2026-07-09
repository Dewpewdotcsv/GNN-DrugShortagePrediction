import pandas as pd
import matplotlib.pyplot as plt
import os

def update_plot():
    df = pd.read_csv("results/model_comparison.csv")
    
    # Plotting
    ax = df.set_index("Model")[["ROC-AUC", "PR-AUC"]].plot(kind="bar", figsize=(12, 7))
    plt.title("Model Performance Comparison (Finalized Results)")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adding text labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig("results/comparison_plot.png")
    print("Successfully updated results/comparison_plot.png")

if __name__ == "__main__":
    update_plot()
