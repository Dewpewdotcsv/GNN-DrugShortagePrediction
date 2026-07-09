import pandas as pd
import numpy as np
import os

DATA_DIR = ""
INPUT_FILE = os.path.join(DATA_DIR, "data/processed/dispensed_l4_aggregated.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "data/processed/supervised_dataset.csv")

def create_features():
    print("Loading aggregated L4 data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run preprocessing.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values(['atc_l4', 'month'])
    
    df['shortage_next'] = df.groupby('atc_l4')['shortage'].shift(-1)
    
    df = df.dropna(subset=['shortage_next'])
    
    df['onset_next'] = ((df['shortage'] == 0) & (df['shortage_next'] == 1)).astype(int)
    
    print("Label distribution:")
    print(df['onset_next'].value_counts())
    
    print("Generating lag features...")
    
    df['shortage_lag1'] = df.groupby('atc_l4')['shortage'].shift(1)
    
    df['shortage_roll3'] = df.groupby('atc_l4')['shortage'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    print("Generating neighbor features...")
    df['atc_l3'] = df['atc_l4'].str[:4]
    
    l3_stats = df.groupby(['atc_l3', 'month'])['shortage'].agg(['sum', 'count']).reset_index()
    l3_stats.rename(columns={'sum': 'l3_sum', 'count': 'l3_count'}, inplace=True)
    
    df = pd.merge(df, l3_stats, on=['atc_l3', 'month'], how='left')
    
    df['neighbor_shortage_mean'] = (df['l3_sum'] - df['shortage']) / (df['l3_count'] - 1)
    df['neighbor_shortage_mean'] = df['neighbor_shortage_mean'].fillna(0.0)
    
    df.drop(columns=['l3_sum', 'l3_count'], inplace=True)
    
    df = df.dropna() 
    
    outfile_neighbor = os.path.join(DATA_DIR, "data/processed/baseline_with_neighbor.csv")
    df.to_csv(outfile_neighbor, index=False)
    print(f"Saved baseline (with neighbor) to {outfile_neighbor}")
    
    if 'neighbor_shortage_mean' in df.columns:
        df_base = df.drop(columns=['neighbor_shortage_mean'])
    else:
        df_base = df.copy()
        
    outfile_base = os.path.join(DATA_DIR, "data/processed/baseline_no_graph.csv")
    df_base.to_csv(outfile_base, index=False)
    print(f"Saved baseline (no neighbor) to {outfile_base}")
    
if __name__ == "__main__":
    create_features()