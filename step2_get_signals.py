import torch
import re
from nnsight import LanguageModel
from loading_utils import load_saes_and_submodules

def extract_feature_index(item):
    """
    Helper to extract a single integer feature index from various container types
    (scalars, 0-d tensors, 1-d tensors, tuples).
    """
    # specific handling for Tensors
    if isinstance(item, torch.Tensor):
        if item.numel() == 1:
            return int(item.item())
        else:
            # If it's a multi-element tensor (e.g. [batch, feature_idx]), 
            # the feature index is typically the last dimension.
            return int(item[-1].item())
    
    # handling for lists/tuples
    if isinstance(item, (list, tuple)):
        return int(item[-1])
    
    # default scalar handling
    return int(item)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the Circuit
    circuit_path = "circuits/debug_circuit.pt"
    print(f"Loading circuit from {circuit_path}...")
    try:
        # weights_only=False is required for custom classes like SparseAct
        circuit_data = torch.load(circuit_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Could not find {circuit_path}.")
        return
    except Exception as e:
        print(f"Error loading circuit: {e}")
        return

    if 'nodes' not in circuit_data:
        print("Error: 'nodes' key not found in circuit file.")
        return

    nodes_dict = circuit_data['nodes']
    
    # Map: layer_index -> list of feature indices we want from that layer
    layer_to_features = {}

    # Regex to capture "name_layer" (e.g., "resid_5", "attn_0")
    pattern = re.compile(r"([a-z]+)_(\d+)")

    print("Parsing nodes and extracting feature indices...")
    for node_name, node_value in nodes_dict.items():
        node_str = str(node_name)
        
        # 1. Parse Layer from Key Name
        match = pattern.match(node_str)
        if match:
            layer = int(match.group(2))
            
            # 2. Extract Feature Indices from Value (SparseAct or list)
            try:
                # First, try to get raw iterable
                if hasattr(node_value, 'act_indices'):
                    # Some SparseAct classes store indices directly here
                    raw_items = node_value.act_indices
                elif hasattr(node_value, 'indices'):
                    raw_items = node_value.indices
                elif isinstance(node_value, torch.Tensor):
                    raw_items = node_value.tolist()
                else:
                    # Fallback: iterate over the object itself
                    raw_items = list(node_value)

                # Process items into clean integers
                features = []
                for item in raw_items:
                    try:
                        idx = extract_feature_index(item)
                        features.append(idx)
                    except Exception:
                        continue # Skip items we can't parse
                
            except Exception as e:
                print(f"Warning: Could not process value for {node_name}: {e}")
                continue

            if not features:
                continue

            # Add unique features for this layer
            if layer not in layer_to_features:
                layer_to_features[layer] = []
            
            current_set = set(layer_to_features[layer])
            current_set.update(features)
            layer_to_features[layer] = list(current_set)

    print(f"Targeting layers: {list(layer_to_features.keys())}")
    
    # Print a summary of counts
    for l, feats in layer_to_features.items():
        print(f"  Layer {l}: {len(feats)} features")

    if not layer_to_features:
        print("No valid nodes found to parse. Exiting.")
        return

    # 2. Load Model using NNSight
    model_name = "EleutherAI/pythia-70m-deduped"
    print(f"Loading {model_name} via nnsight...")
    model = LanguageModel(model_name, device_map="auto", dispatch=True)

    # 3. Load SAEs
    max_layer = max(layer_to_features.keys())
    print(f"Loading SAEs up to layer {max_layer}...")
    
    # We load resid dictionaries
    _, dictionaries = load_saes_and_submodules(
        model, 
        thru_layer=max_layer, 
        include_embed=False, 
        neurons=False, 
        device=device
    )

    # 4. Prepare Inputs
    text = "A B C " * 50
    
    # 5. Extraction using NNSight Trace
    results = {}
    
    print("Running extraction trace...")
    with model.trace(text):
        for layer, features in layer_to_features.items():
            # Find the Submodule object for this layer
            target_submod = None
            target_sae = None
            
            # Look for resid_{layer} in the loaded dictionaries
            for submod, sae in dictionaries.items():
                if submod.name == f"resid_{layer}":
                    target_submod = submod
                    target_sae = sae
                    break
            
            if target_submod is None:
                print(f"Warning: Could not find SAE for resid_{layer}")
                continue
            
            if not features:
                continue
            
            # Convert indices to LongTensor for slicing
            indices = torch.tensor(features, device=device, dtype=torch.long)
                
            # Get activation (Proxy)
            dense_acts = target_submod.get_activation()
            
            # Encode (Proxy computation)
            feature_acts = target_sae.encode(dense_acts)
            
            # Slice specific features (Proxy slicing)
            # feature_acts is [batch, seq, d_sae]
            selected_signals = feature_acts[..., indices]
            
            # Save the result
            results[f"resid_post_layer_{layer}"] = {
                "feature_indices": features,
                "values": selected_signals.save()
            }

    # 6. Post-processing and Saving
    final_output = {}
    for key, data in results.items():
        try:
            final_output[key] = {
                "feature_indices": data["feature_indices"],
                "values": data["values"].value.cpu()
            }
        except Exception as e:
            print(f"Error saving results for {key}: {e}")

    output_path = "signals.pt"
    print(f"Saving extracted signals to {output_path}...")
    torch.save(final_output, output_path)
    print("Done.")

if __name__ == "__main__":
    main()