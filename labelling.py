import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq

def run_fft_labeling():
    # 1. Load Data
    signals_path = "signals.pt"
    if not os.path.exists(signals_path):
        print(f"❌ {signals_path} missing. Run step2_get_signals.py first.")
        return

    print(f"Loading signals from {signals_path}...")
    data = torch.load(signals_path)
    
    output_dir = "final_results"
    os.makedirs(output_dir, exist_ok=True)
    
    final_stats = []
    
    print("Running Spectral Analysis (Normalized)...")
    
    # 2. Iterate through layers
    for layer_key, content in data.items():
        layer_num = int(layer_key.split('_')[-1])
        indices = content['feature_indices']
        
        # Detach and convert to numpy safely
        tensor_data = content['values'][0].float()
        if hasattr(tensor_data, 'detach'):
            tensor_data = tensor_data.detach()
        signal_matrix = tensor_data.numpy()
        
        seq_len, num_feats = signal_matrix.shape
        
        for i, feat_id in enumerate(indices):
            signal = signal_matrix[:, i]
            
            # --- NORMALIZATION FIX ---
            # Scale signal to 0-1 range so "quiet" features can still show patterns
            sig_max = np.max(signal)
            if sig_max > 1e-6:
                signal_norm = signal / sig_max
            else:
                signal_norm = signal # Too small to normalize (dead feature)
            
            # Center it (remove DC offset)
            signal_centered = signal_norm - np.mean(signal_norm)
            
            # FFT
            N = len(signal_norm)
            yf = fft(signal_centered)
            xf = fftfreq(N, 1)[:N//2]
            amplitude = 2.0/N * np.abs(yf[0:N//2])
            
            # Stats
            if np.sum(amplitude) > 0:
                idx_max = np.argmax(amplitude)
                dom_freq = xf[idx_max]
                power = amplitude[idx_max]
            else:
                dom_freq = 0
                power = 0
            
            # Labeling (Adjusted for normalized power)
            # Power is now 0.0 to 1.0 roughly
            label = "Unsure"
            if dom_freq > 0.15: # Fast rhythm
                label = "Induction (High-Freq)"
            elif dom_freq < 0.05 and power > 0.1: # Slow/Static
                label = "Semantic (Static)"
            elif power > 0.1: # Mid-freq
                label = "Rhythmic (Mid-Freq)"
            else:
                label = "Noise/Weak"

            final_stats.append({
                "Layer": layer_num,
                "Feature_ID": feat_id,
                "Dominant_Freq": round(dom_freq, 3),
                "Norm_Power": round(power, 4),
                "Raw_Max": round(sig_max, 4),
                "Auto_Label": label
            })
            
            # Save Plot (Proof)
            if power > 0.1: # Plot anything with a pulse
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(signal, color='dodgerblue')
                plt.title(f"L{layer_num} F{feat_id}: Signal (Max={sig_max:.4f})")
                plt.xlabel("Tokens")
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.bar(xf, amplitude, width=0.01, color='coral')
                plt.title(f"Spectrum (Freq={dom_freq:.2f})")
                plt.xlabel("Frequency")
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/L{layer_num}_F{feat_id}.png")
                plt.close()

    # 3. Save
    df = pd.DataFrame(final_stats)
    df = df.sort_values(by=['Layer', 'Feature_ID'])
    csv_path = f"{output_dir}/automated_labels.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"✅ DONE. Analyzed {len(df)} features.")
    print(f"📂 Plots saved in: {output_dir}/")
    print("="*50)
    print("Sample Labels:")
    print(df[['Layer', 'Feature_ID', 'Raw_Max', 'Auto_Label']].head(10))

if __name__ == "__main__":
    run_fft_labeling()