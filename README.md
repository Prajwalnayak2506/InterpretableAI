# Automated Sparse Feature Circuit Discovery & Labeling

This repository implements an end-to-end pipeline for discovering, extracting, and automatically labeling **sparse feature circuits** in Large Language Models (LLMs).

We introduce a novel methodology combining **Fast Attribution Patching** with **Spectral Analysis (FFT)** to identify and categorize functional components (e.g., *Induction Heads* vs. *Semantic Features*) **without manual inspection**.

---

## 📂 Project Structure

```
├── src/
│   ├── circuit_fast.py       # Step 1: Discover the circuit using IG
│   ├── step2_get_signals.py  # Step 2: Extract time-series activations
│   ├── step3_analysis.py     # Step 3: Generate topology statistics (CSV)
│   ├── step4_fft_labeling.py # Step 4: Run FFT to label features
│   └── visualization.py      # Step 5: Publication-ready plots
├── experiments/              # Runner scripts for different tasks
├── circuits/                 # Saved .pt circuit files
└── results/                  # Final CSVs and figures
```

---

## 🚀 Quick Start

### **1. Installation**

```bash
pip install -r requirements.txt
```

### **2. Circuit Discovery (The Heavy Lifting)**

Run the fast-path attribution patching to find the causal circuit for Subject–Verb Agreement.

```bash
python src/circuit_fast.py
```

**Output:** `circuits/debug_circuit.pt` → Contains the sparse subgraph of discovered features.

---

### **3. Signal Extraction & Probing**

We support multiple probing experiments to test robustness.

#### **Experiment A: Standard Plurality Probe**

```bash
python src/step2_get_signals.py --output signals_plural.pt
```

#### **Experiment B: Gender Agreement Probe (Generalization Test)**

```bash
python src/step2_get_signals.py \
  --text "The boy washes his face. The girl washes her face." \
  --output signals_gender.pt
```

---

### **4. Analysis & Visualization**

Generate spectral labels and paper-ready plots.

#### **Label the features (Induction vs. Semantic)**

```bash
python src/step4_fft_labeling.py
```

#### **Generate Demo Plots
