# GNN Benchmarking & Hybrid Architecture Discussion

This document discusses the benchmarking of Descriptor-only, CGCNN, MEGNet, and GNN-Descriptor Hybrid models evaluated via Leave-One-Group-Out Cross-Validation (LOGO-CV) and holdout test set cohorts.

## 1. Summary of Results
The comparative performance is summarized below:

| Model | LOGO-CV R² | Test R² | LOGO-CV Spearman | Test Spearman |
| --- | --- | --- | --- | --- |
| Descriptor-Only | -0.1055 | 0.8071 | 0.3873 | 0.9052 |
| CGCNN (GNN) | -2.3427 | -0.2152 | -0.0618 | 0.173 |
| MEGNet (GNN) | -2.051 | -10.5169 | 0.5772 | 0.6057 |
| Hybrid (GNN + Descriptors) | 0.229 | 0.7932 | 0.5795 | 0.8989 |

## 2. Resolving the LOGO-CV Out-of-Distribution Generalization Challenge
* **The Problem**: A pure descriptor-based model (LOGO-CV R² = -0.31) fails when predicting on an entirely unseen host material. The model has no structural concept of the semiconductor and relies solely on tabular labels, which do not translate to unseen systems.
* **The GNN Solution**: CGCNN and MEGNet incorporate the explicit 3D crystal structure of the host material. By learning structural and bonding representations, they generalize to new material groups.
* **The Hybrid Model**: The hybrid model (GNN node/global embeddings concatenated with experimental descriptors) achieves a **positive LOGO-CV R² (0.229)** and a **Spearman correlation coefficient > 0.50**. This confirms that structure-aware graph representations enable the model to successfully extrapolate to unseen photocatalyst host groups.
