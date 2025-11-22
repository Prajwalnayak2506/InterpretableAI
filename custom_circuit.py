import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
import torch as t

# Import utilities from the repo
from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit_posaligned
from dictionary_loading_utils import load_saes_and_submodules
from nnsight import LanguageModel
from coo_utils import sparse_reshape

# ==============================================================================
# 🎮 USER ZONE: EDIT YOUR SENTENCES HERE
# ==============================================================================

# SCENARIO: Subject-Verb Agreement
# We want to see how the model knows "The keys" takes "are" and "The key" takes "is".

# 1. The "Correct" Sentence (Clean)
CLEAN_PROMPT = "The keys on the table" 
CLEAN_ANSWER = " are"  # The token the model SHOULD predict

# 2. The "Corrupted" Sentence (Patch) - Change the critical info (e.g., singular/plural)
PATCH_PROMPT = "The key on the table"
PATCH_ANSWER = " is"   # The token the model would predict for the corrupted input

# ==============================================================================

# --- REUSED FAST LOGIC (Do not edit below unless you know what you are doing) ---
def get_circuit_fast(clean, patch, model, embed, attns, mlps, resids, dictionaries, metric_fn, metric_kwargs=dict(), aggregation="sum", nodes_only=False, parallel_attn=False, node_threshold=0.1):
    all_submods = ([embed] if embed is not None else []) + [submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods]
    
    # Fast Forward Pass (1 step IG)
    print("   Calculating Feature Importance...")
    effects, deltas, grads, total_effect = patching_effect(
        clean, patch, model, all_submods, dictionaries, metric_fn,
        metric_kwargs=metric_kwargs, method="ig", steps=1 
    )

    features_by_submod = {submod: effects[submod].abs() > node_threshold for submod in all_submods}
    n_layers = len(resids)

    # Build Nodes Dictionary (With 'y' LAST for plotting safety)
    nodes = {}
    if embed is not None: nodes["embed"] = effects[embed]
    for i in range(n_layers):
        nodes[f"attn_{i}"] = effects[attns[i]]
        nodes[f"mlp_{i}"] = effects[mlps[i]]
        nodes[f"resid_{i}"] = effects[resids[i]]
    nodes["y"] = total_effect

    if nodes_only: return nodes, None

    # Backward Pass (Edges)
    edges = defaultdict(dict)
    edges[f"resid_{len(resids) - 1}"] = {"y": effects[resids[-1]].to_tensor().flatten().to_sparse()}

    def N(upstream, downstream, midstream=[]):
        return jvp(clean, model, dictionaries, downstream, features_by_submod[downstream], upstream, grads[downstream], deltas[upstream], intermediate_stopgrads=midstream)

    print("   Tracing Connections...")
    for layer in reversed(range(len(resids))):
        resid, mlp, attn = resids[layer], mlps[layer], attns[layer]
        edges[f"mlp_{layer}"][f"resid_{layer}"] = N(mlp, resid)
        edges[f"attn_{layer}"][f"resid_{layer}"] = N(attn, resid, [mlp])
        if not parallel_attn: edges[f"attn_{layer}"][f"mlp_{layer}"] = N(attn, mlp)

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

    # Reformat for Plotter
    for child in edges:
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            if parent == "y": edges[child][parent] = sparse_reshape(edges[child][parent], (bc, sc, fc + 1))

    # Aggregation (None)
    for child in edges:
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            if parent == "y": weight = sparse_reshape(edges[child][parent], (bc, sc, fc + 1)).sum(dim=0) / bc
            else: weight = edges[child][parent].sum(dim=(0, 3)) / bc
            edges[child][parent] = weight
    for node in nodes:
        nodes[node] = nodes[node].mean(dim=0)

    return nodes, dict(edges)

class PlotWrapper:
    def __init__(self, tensor):
        self.act = tensor
        self.to_tensor = lambda: self.act

if __name__ == "__main__":
    print(f"🚀 Running Custom Experiment: '{CLEAN_PROMPT}' vs '{PATCH_PROMPT}'")
    
    device = "cuda" if t.cuda.is_available() else "cpu"
    dtype = t.float32
    model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map=device, torch_dtype=dtype)
    
    print("📚 Loading Dictionaries...")
    submodules, dictionaries = load_saes_and_submodules(model, separate_by_type=True, include_embed=True, neurons=False, device=device, dtype=dtype)

    # Tokenize Inputs
    clean_input_ids = model.tokenizer(CLEAN_PROMPT, return_tensors="pt").input_ids.to(device)
    clean_answer_id = model.tokenizer(CLEAN_ANSWER).input_ids[-1]
    patch_input_ids = model.tokenizer(PATCH_PROMPT, return_tensors="pt").input_ids.to(device)
    patch_answer_id = model.tokenizer(PATCH_ANSWER).input_ids[-1]

    # Metric: Logit Diff (How much does the circuit distinguish the two answers?)
    def metric_fn(model, **kwargs):
        logits = model.output.logits[:, -1, :]
        return (logits[:, patch_answer_id] - logits[:, clean_answer_id])

    # Run Discovery
    nodes, edges = get_circuit_fast(
        [CLEAN_PROMPT], [PATCH_PROMPT], model, 
        submodules.embed, submodules.attns, submodules.mlps, submodules.resids, dictionaries, 
        metric_fn, nodes_only=False, aggregation="none", node_threshold=0.1, parallel_attn=True
    )

    # Save Plot
    print("🎨 Drawing Graph...")
    plot_nodes = {k: PlotWrapper(v) if not hasattr(v, 'act') else v for k, v in nodes.items()}
    
    # Load Annotations if available
    annotations = None
    if os.path.exists("annotations/pythia-70m-deduped.jsonl"):
        annotations = {}
        with open("annotations/pythia-70m-deduped.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                if "Annotation" in data: annotations[data["Name"]] = data["Annotation"]

    plot_circuit_posaligned(
        plot_nodes, edges, layers=6, example_text=CLEAN_PROMPT,
        node_threshold=0.1, edge_threshold=0.01, pen_thickness=1,
        annotations=annotations, save_dir="circuits/figures/custom_plot", gemma_mode=False, parallel_attn=True
    )
    print("✅ Done! Check circuits/figures/custom_plot.png")