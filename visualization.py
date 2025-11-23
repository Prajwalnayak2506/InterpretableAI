import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from transformers import AutoTokenizer

# --- HARDCODED DEFAULTS ---
DEFAULT_TEXTS = {
    "signals_plural.pt": "The key is on the table. The keys are on the table. The key is on the table.",
    "signals_gender.pt": "The boy washes his face. The girl washes her face. The boy washes his face."
}

def create_plots():
    parser = argparse.ArgumentParser(description="Generate plots from extracted signals.")
    parser.add_argument("--file", type=str, default="signals.pt", help="Path to the signals .pt file")
    parser.add_argument("--title", type=str, default=None, help="Title prefix for plots")
    parser.add_argument("--text", type=str, default=None, help="The text input used for the probe")
    parser.add_argument("--output_dir", type=str, default="demo_visuals", help="Directory to save plots")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        return
        
    data = torch.load(args.file)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- AUTO-DETECT TEXT & TITLE ---
    filename = os.path.basename(args.file)
    
    # 1. Auto-Fill Text
    if args.text is None:
        if filename in DEFAULT_TEXTS:
            print(f"✨ Auto-detected text for {filename}")
            args.text = DEFAULT_TEXTS[filename]
        else:
            print("⚠️ No text provided. Axis will use numbers.")
    
    # 2. Auto-Fill Title
    if args.title is None:
        if "gender" in filename:
            args.title = "Gender Agreement"
        elif "plural" in filename:
            args.title = "Subject-Verb Agreement"
        else:
            args.title = "Feature Response"

    # Load Tokenizer
    print("⏳ Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    
    token_labels = []
    if args.text:
        tokens = tokenizer(args.text)["input_ids"]
        token_labels = [tokenizer.decode([t]).replace(" ", "") for t in tokens]

    print(f"🎨 Generating plots from {args.file}...")

    all_features = []
    for layer_key, content in data.items():
        vals = content['values'][0].float()
        if hasattr(vals, 'detach'): vals = vals.detach()
        vals = vals.numpy()
        indices = content['feature_indices']
        for i, feat_idx in enumerate(indices):
            all_features.append({'layer': layer_key, 'id': feat_idx, 'signal': vals[:, i]})

    if not all_features:
        print("❌ No features found.")
        return

    # Find top 3 active features
    top_features = sorted(all_features, key=lambda x: np.max(x['signal']), reverse=True)[:3]

    # Plotting
    plt.style.use('default') 

    for i, feat in enumerate(top_features):
        signal = feat['signal']
        
        # Determine Snippet Length based on available labels
        # We want to show as much as we have labels for, or 30, whichever is smaller
        if token_labels:
            snippet_len = min(len(token_labels), len(signal), 30)
            display_labels = token_labels[:snippet_len]
        else:
            snippet_len = min(len(signal), 30)
            display_labels = np.arange(snippet_len)

        # Slice signal to match labels EXACTLY
        viz_signal = signal[:snippet_len]
        x = np.arange(snippet_len)

        # Normalize
        if np.max(viz_signal) > 0:
            viz_signal = (viz_signal - np.min(viz_signal)) / (np.max(viz_signal) - np.min(viz_signal))
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        color = ['#2563EB', '#D946EF', '#10B981'][i % 3]
        
        ax.plot(x, viz_signal, color=color, linewidth=2.5, label=f"Feature {feat['id']}")
        ax.fill_between(x, viz_signal, color=color, alpha=0.15)
        
        peaks = np.where(viz_signal > 0.8)[0]
        if len(peaks) > 0:
            ax.scatter(peaks, viz_signal[peaks], color='#1F2937', s=30, zorder=5)

        layer_name = feat['layer'].split('_')[-1]
        ax.set_title(f"{args.title}: Feature {feat['id']} (Layer {layer_name})", fontsize=14, fontweight='bold')
        
        # Set X-Axis to Tokens (Use 'x' as the ticks locations)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=10)
        
        ax.set_ylabel("Normalized Activation")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        
        clean_filename = args.file.replace(".pt", "").replace("signals_", "")
        save_path = f"{args.output_dir}/{clean_filename}_plot_{i+1}.png"
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    create_plots()