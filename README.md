# Permutation Alignment and Repair Close the Merging Gap in Low-Resource Regimes

Project for **Deep Learning & Applied AI** (DLAI), a.y. 2025/26, Sapienza University of Rome.

## Motivation

Weight averaging of independently trained networks works when the models lie in the same linear-mode-connected (LMC) basin, a condition that holds with shared initialisation and full data, but breaks as soon as each model sees only a fraction of the training set. This regime is directly relevant to federated and distributed learning, where data is partitioned across clients. 
The project asks: *can a lightweight post-hoc alignment step restore the merging benefit?*

## Experimental design

Five ResNet-20 models share the same initialisation but are each trained on a disjoint 10k-sample slice of CIFAR-10 (5 × 10k = 50k total). This setup isolates the effect of **data scarcity** on LMC: the shared initialisation should keep models geometrically close, yet we measure pairwise test-loss barriers 30–50× larger than the same-init/full-data reference, confirming LMC failure is driven by data partitioning, not initialisation diversity (validated explicitly by comparing settings E1A/E1B/E1D).

Standard merging methods (uniform soup, greedy soup, TIES) all collapse to near-random or degenerate to a single-model result. The project then applies a two-step recovery pipeline:

1. **Activation Matching** (Ainsworth et al. 2023) — iterative Hungarian assignment on channel-wise activation correlations resolves the channel-permutation ambiguity of CNNs and collapses the artificial LMC barrier.
2. **Repair** (Jordan et al. 2023) — two epochs of low learning rate SGD on the merged model corrects the residual BatchNorm feature scale drift.

The pipeline is evaluated on five pairs (one different init, four low resource), achieving **76.5%** and **68.2 ± 1.0%** midpoint accuracy respectively, recovering more than 80% of the random-to-endpoint gap. A data-fraction study (1k–10k samples per model) shows the pipeline remains effective across all regimes and at 1k samples per model even surpasses the greedy single-model baseline.

## Additional findings

- **TIES failure mode**: the default λ=1.0 is miscalibrated for this regime, task-vector norms are 0.93–1.44× the base-model norm, far larger than in fine-tuning settings. With λ=0.3, TIES recovers 77.0%, matching or exceeding individual model accuracy.
- **Cosine mergeability metric**: pairwise cosine similarity of task vectors does not predict accuracy drop after merging (Pearson r = +0.28, p = 0.43, n = 10). The metric saturates in this regime, all pairs cluster in a narrow range regardless of cosine.
- **Repair epoch ablation**: accuracy continues to improve beyond 2 epochs without a clear plateau, suggesting the two-epoch result is a conservative lower bound.
