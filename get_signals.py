import torch
import re
import os
import argparse
from nnsight import LanguageModel
from loading_utils import load_saes_and_submodules

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Extract SAE feature activations for a given probe text.")
    parser.add_argument("--text", type=str, default=None, help="Probe text to run through the model")
    parser.add_argument("--output", type=str, default="signals.pt", help="Output filename for the signals")
    parser.add_argument("--circuit", type=str, default="circuits/debug_circuit.pt", help="Path to circuit file")
    args = parser.parse_args()

    # Default probe if none provided
    if args.text is None:
        # Standard robust probe (Singular/Plural mix)
        args.text = "The key is on the table. The keys are on the table. " * 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the Circuit
    print(f"Loading circuit from {args.circuit}...")
    try:
        circuit_data = torch.load(args.circuit, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Could not find {args.circuit}.")
        return

    if 'active_features' in circuit_data:
        print("✅ Found 'active_features' key.")
        layer_to_features_map = circuit_data['active_features']
    else:
        print("❌ Error: 'active_features' list missing from circuit file.")
        return
    
    # Convert layer names to integers
    layer_to_features = {}
    pattern = re.compile(r"([a-z]+)_(\d+)")
    
    for key, indices in layer_to_features_map.items():
        match = pattern.match(str(key))
        if match:
            layer_idx = int(match.group(2))
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            layer_to_features[layer_idx] = indices

    # 3. Load Model
    model_name = "EleutherAI/pythia-70m-deduped"
    print(f"Loading {model_name}...")
    model = LanguageModel(model_name, device_map="auto", dispatch=True)

    # 4. Load SAEs
    max_layer = max(layer_to_features.keys())
    print(f"Loading SAEs up to layer {max_layer}...")
    
    _, dictionaries = load_saes_and_submodules(
        model, thru_layer=max_layer, include_embed=False, neurons=False, device=device
    )

    # 5. Run Trace
    print(f"Input Probe ({len(args.text)} chars): '{args.text[:40]}...'")
    results = {}
    
    print("Running extraction trace...")
    with model.trace(args.text):
        for layer, features in layer_to_features.items():
            target_submod = None
            target_sae = None
            
            for submod, sae in dictionaries.items():
                if f"resid" in submod.name and str(layer) in submod.name:
                    target_submod = submod
                    target_sae = sae
                    break
            
            if target_submod is None or not features: continue
            
            # Filter Error Terms (Index >= 32768)
            valid_features = [int(f) for f in features if int(f) < 32768]
            if not valid_features: continue

            indices = torch.tensor(valid_features, device=device, dtype=torch.long)
            
            dense_acts = target_submod.get_activation()
            feature_acts = target_sae.encode(dense_acts)
            selected_signals = feature_acts[..., indices]
            
            results[f"resid_post_layer_{layer}"] = {
                "feature_indices": valid_features,
                "values": selected_signals.save()
            }

    # 6. Save
    final_output = {}
    for key, data in results.items():
        final_output[key] = {
            "feature_indices": data["feature_indices"],
            "values": data["values"].value.cpu()
        }

    torch.save(final_output, args.output)
    print(f"✅ Saved signals to {args.output}")

if __name__ == "__main__":
    main()