# Personalized Federated Learning on Non-IID Data

> Comparing FedAvg, FedAvg+Fine-Tune, and FedBN on CIFAR-10 with Dirichlet label-skew partitioning — and finding that the "right" personalization strategy depends on *what kind* of heterogeneity you have.

---

## Hypothesis

Federated learning with non-IID data produces a global model that is a poor fit for individual clients. Personalization strategies — specifically local fine-tuning (FedAvg+FT) and local BatchNorm (FedBN) — should recover that lost per-client accuracy. FedBN, which requires no extra communication rounds, was hypothesized to close the gap at lower cost than fine-tuning.

---

## Methods

| Method | Core Idea | Extra Communication |
|---|---|---|
| **FedAvg** | Standard global model, no personalization | Baseline |
| **FedAvg+FT** | Train globally with FedAvg, then fine-tune each client locally | None (post-training only) |
| **FedBN** | Share all weights except BatchNorm layers, which stay local | Slightly less (BN excluded) |

**Data:** CIFAR-10, partitioned across 10 clients using a Dirichlet distribution with concentration parameter α. Low α → highly skewed (each client dominated by 1–2 classes). High α → near-IID.

**Model:** 3-block CNN with BatchNorm after every conv and FC layer.

**Config:** 50 communication rounds · 5 local epochs · 10 clients · batch size 64 · SGD lr=0.01

**Hardware:** NVIDIA T4 GPU (Google Colab)

---

## Results (α = 0.3)

| Method | Mean Acc | Std | Min | Max |
|---|---|---|---|---|
| FedAvg | 0.5972 | 0.1626 | 0.2566 | 0.7702 |
| FedAvg+FT | **0.9482** | **0.0199** | 0.9032 | 0.9692 |
| FedBN | 0.3559 | 0.1419 | 0.1030 | 0.5617 |

### Per-Client Accuracy
![Per-client comparison](plots/per_client_comparison.png)

FedAvg+FT lifts every client above 0.90. FedBN consistently falls below FedAvg, with some clients below 0.15.

### Distribution of Per-Client Accuracy
![Boxplot](plots/accuracy_boxplot.png)

FedAvg+FT has the highest median and tightest spread — the most equitable outcome across clients. FedBN has both lower median and higher variance than FedAvg.

### Global Test Accuracy over Rounds
![Learning curves](plots/learning_curves.png)

FedAvg converges to ~0.60 global accuracy. FedBN's global accuracy stays low (~0.30) because the server's BatchNorm parameters are never updated — this is by design, not a failure.

---

## Finding: FedBN Underperforms on Label-Shift Non-IID

**FedBN was designed for domain/feature shift** — scenarios where clients have different image statistics (different scanners, cameras, domains). In that setting, local BatchNorm statistics capture meaningful per-client feature distributions.

**CIFAR-10 with Dirichlet label skew is a label-shift problem**, not a feature-shift problem. All clients see the same type of CIFAR-10 images; only the class proportions differ. Local BatchNorm statistics offer no advantage here. Meanwhile, the shared backbone receives contradictory gradient signals from clients with wildly different class distributions, leading to poor convergence of the non-BN parameters.

FedAvg+FT wins because it directly adapts the classifier to each client's local class prior — which is exactly the right fix for label shift.

### Alpha Sensitivity
![Alpha sensitivity](plots/alpha_sensitivity.png)

The finding holds at every heterogeneity level. Across α ∈ {0.1, 0.3, 0.5, 1.0, 5.0}:
- FedAvg+FT stays flat at ~0.94 — insensitive to how skewed the data is
- FedAvg stays flat at ~0.60–0.63
- FedBN stays flat and below FedAvg at every alpha value

**Personalization strategy choice depends on the type of heterogeneity, not just the degree.**

---

## Reproducing the Results

**Runtime: ~20 minutes on Google Colab T4 GPU (free tier)**

**Step 1 — Open a new Colab notebook and upload all `.py` files:**
```python
from google.colab import files
files.upload()
# Select: config.py, data.py, model.py, client.py, server.py, train.py, plot.py, run.py
```

**Step 2 — Install dependencies:**
```python
!pip install torch torchvision matplotlib seaborn numpy -q
```

**Step 3 — Run:**
```python
%run run.py
```

Plots are automatically saved to `MyDrive/FL_Project/plots/` in your Google Drive.

### Key config parameters (`config.py`)
| Parameter | Default | Effect |
|---|---|---|
| `DIRICHLET_ALPHA` | 0.3 | Lower → more non-IID |
| `NUM_ROUNDS` | 50 | More rounds → better convergence |
| `LOCAL_EPOCHS` | 5 | More epochs → more client drift |
| `NUM_CLIENTS` | 10 | Number of federated clients |

---

## Project Structure

```
├── config.py      # All hyperparameters
├── data.py        # CIFAR-10 download + Dirichlet partitioning
├── model.py       # CNN with BatchNorm layers
├── client.py      # Per-client train / fine-tune / evaluate
├── server.py      # FedAvg and FedBN aggregation
├── train.py       # Experiment runners
├── plot.py        # Visualization
└── run.py         # Entry point
```

---

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 — **FedAvg**
- Li et al., *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization*, ICLR 2021 — **FedBN**
- Zhao et al., *Federated Learning with Non-IID Data*, arXiv 2018 — **Non-IID analysis**
"# FedPersona" 
"# FedPersona" 
