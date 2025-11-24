import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict

import torch
import torch as t 

from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_loading_utils import load_saes_and_submodules
from nnsight import LanguageModel
from data_loading_utils import load_examples
from coo_utils import sparse_reshape

def get_circuit_diagnostic(
    clean,
    patch,
    model,
    embed,
    attns,
    mlps,
    resids,
    dictionaries,
    metric_fn,
    metric_kwargs=dict(),
    aggregation="sum",
    nodes_only=False,
    parallel_attn=False,
):
    all_submods = ([embed] if embed is not None else []) + [
        submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods
    ]

    print("   DEBUG: Running 1-Step Patching (IG)...")
    
    effects, deltas, grads, total_effect = patching_effect(
        clean, patch, model, all_submods, dictionaries, metric_fn,
        metric_kwargs=metric_kwargs, method="ig", steps=1 
    )
    print("   DEBUG: Forward Pass Complete.")

    # 1. DIAGNOSTIC: Check Max Values & Determine Threshold safely
    max_scores = []
    print("   📊 Analyzing Attribution Scores...")
    for i, resid in enumerate(resids):
        eff = effects[resid]
        try:
            # FAIL-SAFE: Convert to dense tensor first
            if hasattr(eff, 'to_tensor'):
                dense_eff = eff.to_tensor()
            else:
                dense_eff = eff
            
            m = dense_eff.abs().max().item()
        except Exception as e:
            # If standard max fails, try a fallback for sparse
            try:
                if hasattr(eff, 'values') and eff.values.numel() > 0:
                    m = eff.values.abs().max().item()
                else:
                    m = 0.0
            except:
                m = 0.0
        
        max_scores.append(m)
        print(f"      Layer {i} Max Score: {m:.6f}")

    # Set Dynamic Threshold: 10% of the global max score
    global_max = max(max_scores) if max_scores else 0
    if global_max == 0:
        print("   ❌ FATAL: All attribution scores are 0.0. Check your Metric function!")
        # Fallback to avoid crash
        return None, None, None, 0.1

    dynamic_threshold = global_max * 0.10
    print(f"   ⚡ Auto-Selected Threshold: {dynamic_threshold:.6f} (10% of max {global_max:.6f})")

    # 2. Filter Features using Dynamic Threshold (Robust Method)
    active_features_dict = {}
    print("   🔍 Extracting Active Features...")
    
    for i, resid in enumerate(resids):
        eff = effects[resid]
        try:
            # Again, convert to dense to be safe
            if hasattr(eff, 'to_tensor'):
                dense_eff = eff.to_tensor().detach().cpu()
            else:
                dense_eff = eff.detach().cpu()
            
            # Find indices where activation > threshold
            # 1. Create boolean mask
            mask = dense_eff.abs() > dynamic_threshold
            
            # 2. Find non-zero indices (Using explicit 'torch' now)
            nonzero_indices = torch.nonzero(mask)
            
            if nonzero_indices.numel() > 0:
                # The feature index is the last column
                feats = nonzero_indices[:, -1].unique().tolist()
                active_features_dict[f"resid_{i}"] = feats
                print(f"      -> Layer {i}: Kept {len(feats)} features")
            else:
                active_features_dict[f"resid_{i}"] = []
                print(f"      -> Layer {i}: 0 features > {dynamic_threshold:.6f}")
                
        except Exception as e:
            print(f"      -> Layer {i} Error: {e}")
            active_features_dict[f"resid_{i}"] = []

    # 3. Standard Circuit Construction
    features_by_submod = {
        submod: effects[submod].abs() > dynamic_threshold for submod in all_submods
    }

    n_layers = len(resids)
    nodes = {} 
    if embed is not None: nodes["embed"] = effects[embed]
    for i in range(n_layers):
        nodes[f"attn_{i}"] = effects[attns[i]]
        nodes[f"mlp_{i}"] = effects[mlps[i]]
        nodes[f"resid_{i}"] = effects[resids[i]]
    nodes["y"] = total_effect

    edges = defaultdict(dict)
    edges[f"resid_{len(resids) - 1}"] = {
        "y": effects[resids[-1]].to_tensor().flatten().to_sparse()
    }

    def N(upstream, downstream, midstream=[]):
        return jvp(
            clean, model, dictionaries, downstream,
            features_by_submod[downstream], upstream,
            grads[downstream], deltas[upstream],
            intermediate_stopgrads=midstream,
        )

    print("   DEBUG: Computing Edges...")
    for layer in reversed(range(len(resids))):
        resid, mlp, attn = resids[layer], mlps[layer], attns[layer]
        edges[f"mlp_{layer}"][f"resid_{layer}"] = N(mlp, resid)
        edges[f"attn_{layer}"][f"resid_{layer}"] = N(attn, resid, [mlp])
        if not parallel_attn:
            edges[f"attn_{layer}"][f"mlp_{layer}"] = N(attn, mlp)

        prev_resid = resids[layer - 1] if layer > 0 else embed
        if prev_resid is not None:
            if layer > 0:
                edges[f"resid_{layer - 1}"][f"mlp_{layer}"] = N(prev_resid, mlp, [attn])
                edges[f"resid_{layer - 1}"][f"attn_{layer}"] = N(prev_resid, attn)
                edges[f"resid_{layer - 1}"][f"resid_{layer}"] = N(prev_resid, resid, [mlp, attn])
            else:
                edges["embed"][f"mlp_{layer}"] = N(prev_resid, mlp, [attn])
                edges["embed"][f"attn_{layer}"] = N(prev_resid, attn)
                edges["embed"]["resid_0"] = N(prev_resid, resid, [mlp, attn])

    # Reformat
    for child in edges:
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            if parent == "y":
                edges[child][parent] = sparse_reshape(edges[child][parent], (bc, sc, fc + 1))

    if aggregation == "none":
        for child in edges:
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                if parent == "y":
                    edges[child][parent] = edges[child][parent].sum(dim=0) / bc
                else:
                    edges[child][parent] = edges[child][parent].sum(dim=(0, 3)) / bc
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    return nodes, dict(edges), active_features_dict, dynamic_threshold

if __name__ == "__main__":
    MODEL_ID = "EleutherAI/pythia-70m-deduped"
    DATASET = "rc_train"
    NUM_EXAMPLES = 5 
    BATCH_SIZE = 1
    
    print("🚀 Starting Diagnostic Circuit Discovery (Version 7 - Import Fix)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    dtype = torch.float32
    model = LanguageModel(MODEL_ID, device_map=device, torch_dtype=dtype)
    
    print("📚 Loading Autoencoders...")
    submodules, dictionaries = load_saes_and_submodules(
        model, separate_by_type=True, include_embed=True, neurons=False, device=device, dtype=dtype,
    )

    data_path = f"data/{DATASET}.json"
    examples = load_examples(data_path, NUM_EXAMPLES, model, use_min_length_only=True)
    batch = examples[:BATCH_SIZE]
    
    clean_inputs = [e["clean_prefix"] for e in batch]
    clean_answer_idxs = torch.tensor([model.tokenizer(e["clean_answer"]).input_ids[-1] for e in batch], dtype=torch.long, device=device)
    patch_inputs = [e["patch_prefix"] for e in batch]
    patch_answer_idxs = torch.tensor([model.tokenizer(e["patch_answer"]).input_ids[-1] for e in batch], dtype=torch.long, device=device)

    def metric_fn(model, **kwargs):
        logits = model.output.logits[:, -1, :]
        return torch.gather(logits, dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
               torch.gather(logits, dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)

    # RUN
    nodes, edges, active_features, final_threshold = get_circuit_diagnostic(
        clean_inputs, patch_inputs, model,
        submodules.embed, submodules.attns, submodules.mlps, submodules.resids, dictionaries,
        metric_fn, nodes_only=False, aggregation="none", 
        parallel_attn=True,
    )

    if nodes is not None:
        print("🎉 Circuit Computed Successfully!")
        
        save_dir = "circuits"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = f"{save_dir}/debug_circuit.pt"
        save_dict = {
            "examples": examples, 
            "nodes": nodes, 
            "edges": edges, 
            "active_features": active_features,
            "used_threshold": final_threshold
        }
        
        torch.save(save_dict, save_path)
        print(f"💾 Saved circuit to: {save_path}")

        print("🎨 Generating Plot...")
        plot_dir = "circuits/figures"
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        
        annotations = None
        if os.path.exists(f"annotations/pythia-70m-deduped.jsonl"):
            annotations = {}
            with open(f"annotations/pythia-70m-deduped.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "Annotation" in data: annotations[data["Name"]] = data["Annotation"]

        class PlotWrapper:
            def __init__(self, tensor):
                self.act = tensor
                self.to_tensor = lambda: self.act
        
        plot_nodes = {k: PlotWrapper(v) if not hasattr(v, 'act') else v for k, v in nodes.items()}

        plot_circuit_posaligned(
            plot_nodes, edges, layers=6, example_text=examples[0]["clean_prefix"],
            node_threshold=final_threshold, 
            edge_threshold=0.01, pen_thickness=1,
            annotations=annotations, save_dir=f"{plot_dir}/debug_plot", gemma_mode=False, parallel_attn=True,
        )
        print(f"🖼️ Plot saved to: {plot_dir}/debug_plot.png")
        print("✅ DONE.")