import torch
import pandas as pd
import os
import re

def extract_results_from_saved_list():
    # 1. Load the file
    circuit_path = "circuits/debug_circuit.pt"
    if not os.path.exists(circuit_path):
        print(f"❌ File missing: {circuit_path}")
        return

    print(f"📂 Opening {circuit_path}...")
    # Force CPU load to avoid any device arguments
    data = torch.load(circuit_path, map_location='cpu', weights_only=False)
    
    # 2. LOOK FOR THE GOLDEN KEY: 'active_features'
    # We saved this explicitly in the last version of circuit_fast.py
    if 'active_features' not in data:
        print("❌ CRITICAL: 'active_features' list is missing from the file.")
        print("   Did you run the V7 'circuit_fast.py' I gave you last time?")
        print("   The log must show: 'Layer X: Kept Y features'")
        return

    print("✅ Found feature list! extracting data...")
    active_features = data['active_features']
    
    # 3. Build the Excel Sheet
    stats_list = []
    # Pattern to match "resid_2", "mlp_4", etc.
    pattern = re.compile(r"([a-z]+)_(\d+)")
    
    total_count = 0
    
    for layer_key, feature_ids in active_features.items():
        # Parse layer name
        match = pattern.match(str(layer_key))
        if not match: continue
        
        node_type = match.group(1) # resid, mlp, attn
        layer_idx = int(match.group(2))
        
        # feature_ids is a list of integers (e.g. [1045, 2099])
        for fid in feature_ids:
            stats_list.append({
                "Layer": layer_idx,
                "Component": node_type,
                "Feature_ID": int(fid),
                "Status": "Active"
            })
            total_count += 1

    if total_count == 0:
        print("⚠️ The file exists, but the feature list is empty.")
        print("   This means the threshold (0.15) was too high.")
        return

    # 4. Save to CSV
    df = pd.DataFrame(stats_list)
    # Sort for prettiness
    df = df.sort_values(by=['Layer', 'Component', 'Feature_ID'])
    
    os.makedirs("analysis_results", exist_ok=True)
    csv_path = "analysis_results/final_circuit_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*40)
    print(f"🎉 SUCCESS! Extracted {total_count} features.")
    print("="*40)
    print(df.groupby(['Layer', 'Component']).count())
    print("="*40)
    print(f"📄 Excel file saved: {csv_path}")
    print("👉 Send this file to your friend for FFT analysis.")

if __name__ == "__main__":
    extract_results_from_saved_list()