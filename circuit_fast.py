import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict

import torch as t
# We import their utilities
from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_learning import AutoEncoder
from data_loading_utils import load_examples, load_examples_nopair
from dictionary_loading_utils import load_saes_and_submodules
from nnsight import LanguageModel
from coo_utils import sparse_reshape

# --- OVERRIDE: Fast Circuit Generation ---
def get_circuit_fast(
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
    node_threshold=0.1,
):
    all_submods = ([embed] if embed is not None else []) + [
        submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods
    ]

    print("   DEBUG: Starting Patching Effect (Forward Pass)...")
    
    # --- FIX 1: Pass steps=1 directly ---
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method="ig", 
        steps=1 
    )
    print("   DEBUG: Forward Pass Complete. Filtering Features...")

    features_by_submod = {
        submod: effects[submod].abs() > node_threshold for submod in all_submods
    }

    n_layers = len(resids)

    # --- FIX 2: Reorder Nodes (Add 'y' LAST) ---
    # The plotter assumes the first node has sequence length. 
    # 'y' is scalar, so if it's first, the plotter crashes.
    nodes = {} 
    
    if embed is not None:
        nodes["embed"] = effects[embed]
    for i in range(n_layers):
        nodes[f"attn_{i}"] = effects[attns[i]]
        nodes[f"mlp_{i}"] = effects[mlps[i]]
        nodes[f"resid_{i}"] = effects[resids[i]]
        
    # Add 'y' last
    nodes["y"] = total_effect

    if nodes_only:
        if aggregation == "sum":
            for k in nodes:
                if k != "y":
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    # --- FIX 3: Use 'dict' (not lambda) for saving ---
    edges = defaultdict(dict)
    
    edges[f"resid_{len(resids) - 1}"] = {
        "y": effects[resids[-1]].to_tensor().flatten().to_sparse()
    }

    def N(upstream, downstream, midstream=[]):
        result = jvp(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            intermediate_stopgrads=midstream,
        )
        return result

    print("   DEBUG: Computing Edges (Backward Pass)...")
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        MR_effect = N(mlp, resid)
        AR_effect = N(attn, resid, [mlp])
        edges[f"mlp_{layer}"][f"resid_{layer}"] = MR_effect
        edges[f"attn_{layer}"][f"resid_{layer}"] = AR_effect

        if not parallel_attn:
            AM_effect = N(attn, mlp)
            edges[f"attn_{layer}"][f"mlp_{layer}"] = AM_effect

        if layer > 0:
            prev_resid = resids[layer - 1]
        else:
            prev_resid = embed

        if prev_resid is not None:
            RM_effect = N(prev_resid, mlp, [attn])
            RA_effect = N(prev_resid, attn)
            RR_effect = N(prev_resid, resid, [mlp, attn])

            if layer > 0:
                edges[f"resid_{layer - 1}"][f"mlp_{layer}"] = RM_effect
                edges[f"resid_{layer - 1}"][f"attn_{layer}"] = RA_effect
                edges[f"resid_{layer - 1}"][f"resid_{layer}"] = RR_effect
            else:
                edges["embed"][f"mlp_{layer}"] = RM_effect
                edges["embed"][f"attn_{layer}"] = RA_effect
                edges["embed"]["resid_0"] = RR_effect

    print("   DEBUG: Reformatting Matrices...")
    for child in edges:
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == "y":
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc + 1))
            else:
                continue
            edges[child][parent] = weight_matrix

    if aggregation == "sum":
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != "y":
                nodes[node] = nodes[node].sum(dim=1)

        for child in edges:
            bc, _ = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, _ = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0, 2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != "y":
                nodes[node] = nodes[node].mean(dim=0)

    elif aggregation == "none":
        for child in edges:
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc + 1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            # Mean converts SparseAct to Tensor
            nodes[node] = nodes[node].mean(dim=0)

    return nodes, dict(edges)

if __name__ == "__main__":
    # HARDCODED DEFAULTS
    MODEL_ID = "EleutherAI/pythia-70m-deduped"
    DATASET = "rc_train"
    NUM_EXAMPLES = 5 
    BATCH_SIZE = 1
    
    print("🚀 Starting Fast Circuit Discovery (Final V4)...")
    
    # 1. Setup Device
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    # 2. Load Model
    print("⏳ Loading Model...")
    dtype = t.float32
    model = LanguageModel(MODEL_ID, device_map=device, torch_dtype=dtype)
    print("✅ Model Loaded.")

    # 3. Load Data
    print("📖 Loading Data...")
    data_path = f"data/{DATASET}.json"
    examples = load_examples(data_path, NUM_EXAMPLES, model, use_min_length_only=True)
    num_examples = min(len(examples), NUM_EXAMPLES)
    
    batch = examples[:BATCH_SIZE]
    print(f"⚡ Processing 1 batch of {BATCH_SIZE} example(s)...")

    # 4. Load Autoencoders
    print("📚 Loading Autoencoders...")
    submodules, dictionaries = load_saes_and_submodules(
        model,
        separate_by_type=True,
        include_embed=True,
        neurons=False,
        device=device,
        dtype=dtype,
    )

    # 5. Prep Inputs
    clean_inputs = [e["clean_prefix"] for e in batch]
    clean_answer_idxs = t.tensor(
        [model.tokenizer(e["clean_answer"]).input_ids[-1] for e in batch],
        dtype=t.long,
        device=device,
    )
    patch_inputs = [e["patch_prefix"] for e in batch]
    patch_answer_idxs = t.tensor(
        [model.tokenizer(e["patch_answer"]).input_ids[-1] for e in batch],
        dtype=t.long,
        device=device,
    )

    # 6. Define Metric
    def metric_fn(model, **kwargs):
        logits = model.output.logits[:, -1, :]
        return t.gather(
            logits, dim=-1, index=patch_answer_idxs.view(-1, 1)
        ).squeeze(-1) - t.gather(
            logits, dim=-1, index=clean_answer_idxs.view(-1, 1)
        ).squeeze(-1)

    # 7. RUN CIRCUIT
    print("▶️ Executing Circuit Logic...")
    nodes, edges = get_circuit_fast(
        clean_inputs,
        patch_inputs,
        model,
        submodules.embed,
        submodules.attns,
        submodules.mlps,
        submodules.resids,
        dictionaries,
        metric_fn,
        nodes_only=False,
        aggregation="none",
        node_threshold=0.1,
        parallel_attn=True,
    )
    print("🎉 Circuit Computed Successfully!")

    # 8. Save Results
    save_dir = "circuits"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = f"{save_dir}/debug_circuit.pt"
    save_dict = {"examples": examples, "nodes": nodes, "edges": edges}
    
    try:
        t.save(save_dict, save_path)
        print(f"💾 Saved circuit to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving (skipping): {e}")

    # 9. Plotting
    print("🎨 Generating Plot...")
    plot_dir = "circuits/figures"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    annotations = None
    if os.path.exists(f"annotations/pythia-70m-deduped.jsonl"):
        annotations = {}
        with open(f"annotations/pythia-70m-deduped.jsonl", "r") as f:
            for line in f:
                line = json.loads(line)
                if "Annotation" in line:
                    annotations[line["Name"]] = line["Annotation"]

    # --- FIX 4: Wrapper for Plotter ---
    class PlotWrapper:
        def __init__(self, tensor):
            self.act = tensor
            # Helper for max/min calculations in to_hex
            self.to_tensor = lambda: self.act
    
    # Wrap raw tensors so the plotter sees .act
    plot_nodes = {
        k: PlotWrapper(v) if not hasattr(v, 'act') else v 
        for k, v in nodes.items()
    }

    plot_circuit_posaligned(
        plot_nodes,
        edges,
        layers=6,
        example_text=examples[0]["clean_prefix"],
        node_threshold=0.1,
        edge_threshold=0.01,
        pen_thickness=1,
        annotations=annotations,
        save_dir=f"{plot_dir}/debug_plot",
        gemma_mode=False,
        parallel_attn=True,
    )
    print(f"🖼️ Plot saved to: {plot_dir}/debug_plot.png")
    print("✅ DONE.")