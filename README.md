# R-DARTS: Recursive Differentiable Architecture Search

> **Status:** Proof of Concept / Experimental  
> **Full Report:** [Read the Master's Thesis (PDF)](./report.pdf) > **Original Concept:** Based on [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (Liu et al., 2018)

## The Concept

**R-DARTS** extends the original DARTS formulation by introducing the concept of **Recursive Search Spaces**.

In standard DARTS, the algorithm searches for a cell structure using a fixed set of primitive operations (Convolution 3x3, MaxPool, etc.). Once found, this cell is stacked to form the final network.

**R-DARTS asks: What if the "learned cell" became a "primitive operation" for a subsequent search?**

### The Recursive Loop

1.  **Level 0 (Standard):** Search for a `Genotype_L0` using atomic primitives (Conv, Pool, Skip).
    - _Command:_ `uv run cnn/train_search.py`
2.  **Abstraction:** Wrap `Genotype_L0` into a `CellOp` — a black-box operation that behaves like a standard PyTorch module but contains the complex topology learned in step 1.
3.  **Level 1 (Recursive):** Launch a new search where the set of primitives is composed of the top-k learned cells (and 'none'):
    $$ \mathcal{O}_{L2} = \{ \text{none}, G_{0}^1, \dots, G\_{0}^k \} $$
    - _Command:_ `uv run cnn/train_search.py --recursive_genotypes "Genotype_L0_v1,Genotype_L0_v2"`
4.  **Result:** A hyper-cell composed of standard operations and nested sub-cells.

This approach mimics biological evolution, where simple cells combine to form tissues, and tissues combine to form organs. By ensuring optimality at the local level (micro-structure) first, R-DARTS accelerates the global search (macro-structure) by assembling pre-optimized components rather than learning from scratch.

## Contributions

Compared to the original DARTS implementation (2018), this repository introduces:

1.  **Recursive Search Engine:**

    - Implementation of `CellOp` (`cnn/operations.py`), allowing any learned graph to be wrapped as a differentiable primitive.
    - Dynamic injection of primitives in `model_search.py` and `train_search.py`, enabling the search space to evolve at runtime.

2.  **Modernized PyTorch Stack:**

    - Migration from legacy PyTorch 0.3 to modern **PyTorch 2.x**.
    - Replacement of deprecated `Variable(volatile=True)` with `torch.no_grad()`.
    - Full type hinting (compliant with `pyright`).

3.  **Reproducibility:**
    - Dependency management via **`uv`** (replacing fragile requirements files).
    - Unified workflow for searching and retraining recursive architectures.

---

## Implementation Details

This repository contains a functional implementation of the **Recursive Graph Construction**.

### Scope of Modernization (CNN Only)

**Important:** Only the **CNN (Convolutional Neural Network)** portion of the original DARTS codebase (`cnn/` folder) has been modernized and adapted for R-DARTS. The `rnn/` (Recurrent Neural Network) folder remains in its legacy state (PyTorch 0.3 era) and has **not** been updated or tested.

### Key Components

- **`cnn/operations.py` -> `CellOp`**: An adapter class that converts a static `Genotype` (a list of connections) into a differentiable `nn.Module` compatible with the DARTS search engine. It handles stride adaptation and channel projection automatically.
- **`cnn/model_search.py`**: Refactored to accept dynamic `primitives` and `ops` dictionaries, breaking the hard-coded dependency on the original 8 primitives.
- **`cnn/train_search.py`**: The main training script, upgraded to support dynamic injection of recursive primitives via command line arguments.

## Usage

### Prerequisites

```bash
uv sync
```

### 1. Classical DARTS Search (Level 0)

First, run a standard architecture search to discover the base building blocks.
_(Currently, this uses the standard DARTS primitives)_

```bash
# Run from project root
uv run cnn/train_search.py --batch_size 64
```

**⚠️ IMPORTANT STEP (where to read results):**

- The script writes logs in `search-*/log.txt`.
- `Best (argmax) Genotype: ...` → deterministic genotype to train/evaluate.
- `Sampled Genotype N: ...` → sampled variants to use as primitives for recursion.

**Copy the chosen genotype(s) into `cnn/genotypes.py`**, giving them names (e.g., `MY_LEVEL0_CELL` for the argmax, `MY_LEVEL0_CELL_V1`, `MY_LEVEL0_CELL_V2` for sampled variants).

Example in `cnn/genotypes.py`:

```python
MY_LEVEL0_CELL = Genotype(
    normal=[('sep_conv_3x3', 1), ...],
    normal_concat=[...],
    reduce=[...],
    reduce_concat=[...]
)
```

### 2. Recursive Search (Level 1)

Now, launch a new search using your previously learned cells as building blocks. The script will look up the variable names you created in `cnn/genotypes.py`. You can provide multiple genotypes (comma-separated) to enrich the search space.

```bash
# Run from project root
uv run cnn/train_search.py \
    --batch_size 16 \
    --recursive_genotypes "MY_LEVEL0_CELL_V1,MY_LEVEL0_CELL_V2" \
    --save search-l1
```

**Note:** Recursive cells are more memory-intensive, so reducing the `batch_size` (e.g. to 16 or 32) is recommended.

### 3. Final Evaluation (Training from Scratch)

Once you have found a recursive architecture (let's call it `MY_LEVEL1_CELL`), copy it into `cnn/genotypes.py`. You can now train it fully for 600 epochs to measure its true performance.

```bash
# Run from project root
uv run cnn/train.py \
    --auxiliary \
    --cutout \
    --arch MY_LEVEL1_CELL \
    --recursive_genotypes "MY_LEVEL0_CELL_V1,MY_LEVEL0_CELL_V2"
```

_Note: If your `MY_LEVEL1_CELL` uses `MY_LEVEL0_CELL_V1` and `MY_LEVEL0_CELL_V2` as primitives, you MUST pass them to `--recursive_genotypes` so the evaluation script knows how to build the graph._

---

## Historical Context & Acknowledgments

This project was originally developed as a **Master's Thesis (2022)**. It explores hierarchical Neural Architecture Search (NAS) by treating learned architectures as reusable primitives.

It has been recently **modernized** (Dec 2025) to run on contemporary PyTorch versions and modern Python environments (using `uv`). While the core recursive logic has been reimplemented and verified as functional, the codebase is provided "as is" to spark discussion on hierarchical NAS.

### Supervision

Special thanks to [Enzo Tartaglione](https://enzotarta.github.io/) (Telecom Paris) for his supervision and guidance during the original development of this thesis.

## Reference

This code is a fork of the official [DARTS implementation](https://github.com/quark0/darts).

```bibtex
@inproceedings{liu2018darts,
  title={DARTS: Differentiable Architecture Search},
  author={Hanxiao Liu and Karen Simonyan and Yiming Yang},
  booktitle={ICLR},
  year={2019}
}
```
