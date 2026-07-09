import pandas as pd
import numpy as np
import os
import re

DATA_DIR = "."
ATC_FILE = os.path.join(DATA_DIR, "WHO ATC-DDD 2024-07-31.csv")
SHORTAGE_FILE = os.path.join(DATA_DIR, "shortage_report_export.csv")
DISPENSED_FILE = os.path.join(DATA_DIR, "data/monthly_dispensed_l4.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "data/processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_atc_code(code):
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    return code if len(code) >= 1 else None

def load_and_process_atc():
    print("Processing ATC dataset...")
    
    df = pd.read_csv(ATC_FILE)
    
    df['atc_code'] = df['atc_code'].apply(clean_atc_code)
    df = df.dropna(subset=['atc_code'])
    
    df['atc_l1'] = df['atc_code'].str[0]
    df['atc_l2'] = df['atc_code'].str[:3]
    df['atc_l3'] = df['atc_code'].str[:4]
    df['atc_l4'] = df['atc_code'].str[:5]
    df['atc_l5'] = df['atc_code'] 
    
    outfile = os.path.join(OUTPUT_DIR, "atc_hierarchy.csv")
    df.to_csv(outfile, index=False)
    print(f"Saved {outfile}")
    return df

def load_and_process_shortages():
    print("Processing Shortage dataset...")
    
    df = pd.read_csv(SHORTAGE_FILE, skiprows=1)
    
    df.columns = [c.lower().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    
    df['actual_start_date'] = pd.to_datetime(df['actual_start_date'], errors='coerce')
    df['actual_end_date'] = pd.to_datetime(df['actual_end_date'], errors='coerce')
    
    df['atc_code'] = df['atc_code'].astype(str).str.strip().str.upper()
    
    df = df[df['atc_code'] != 'NAN']
    
    df['atc_l4'] = df['atc_code'].str[:5]
    
    outfile = os.path.join(OUTPUT_DIR, "shortages_cleaned.csv")
    df.to_csv(outfile, index=False)
    print(f"Saved {outfile}")
    return df

def process_dispensed_data():
    print("Processing Dispensed dataset...")
    df = pd.read_csv(DISPENSED_FILE)
    
    df['atc_l5'] = df['ig_id'].str.replace('IG_', '', regex=False)
    df['atc_l4'] = df['atc_l5'].str[:5]
    
    outfile_l5 = os.path.join(OUTPUT_DIR, "dispensed_l5.csv")
    df.to_csv(outfile_l5, index=False)
    print(f"Saved {outfile_l5}")

    print("Aggregating to L4 level...")
    
    l4_df = df.groupby(['atc_l4', 'month']).agg({
        'patients': 'sum',
        'rx_count': 'sum',
        'days_of_supply': 'sum',
        'dos_per_patient': 'mean', 
        'shortage': 'max' 
                          
    }).reset_index()

    outfile_l4 = os.path.join(OUTPUT_DIR, "dispensed_l4_aggregated.csv")
    l4_df.to_csv(outfile_l4, index=False)
    print(f"Saved {outfile_l4}")
    return l4_df

if __name__ == "__main__":
    load_and_process_atc()
    load_and_process_shortages()
    process_dispensed_data()