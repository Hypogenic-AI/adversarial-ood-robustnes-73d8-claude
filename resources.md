# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Probing the Limits: Adversarial and OOD Robustness of Neural Thicket Ensembles."

## Papers
Total papers downloaded: 18

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Neural Thickets | Gan, Isola | 2026 | papers/gan2026_neural_thickets.pdf | Core paper; RandOpt algorithm |
| 2 | ADP: Ensemble Diversity for Adversarial Robustness | Pang et al. | 2019 | papers/pang2019_adversarial_robustness_ensemble_diversity.pdf | ADP regularizer; ICML |
| 3 | Split-Ensemble: OOD-aware Ensemble | Chen, Yang, Gan et al. | 2024 | papers/chen2023_split_ensemble_ood.pdf | Subtask splitting; ICML |
| 4 | Seasoning Model Soups | Croce et al. | 2023 | papers/croce2023_seasoning_model_soups_robustness.pdf | Weight-space soups; CVPR |
| 5 | Ensemble Diversity for OOD Detection | Xia, Bouganis | 2022 | papers/xia2022_ensemble_diversity_ood_detection.pdf | MI not useful for OOD |
| 6 | Ensemble Adversarial Defense Low Curvature | Zhao et al. | 2024 | papers/zhao2024_ensemble_adversarial_defense_low_curvature.pdf | Dispersed low-curvature models |
| 7 | MEAT: Median-Ensemble Adversarial Training | Hu et al. | 2024 | papers/hu2024_MEAT_median_ensemble_adversarial.pdf | Median weight averaging |
| 8 | Deep Ensembles: Loss Landscape | Fort et al. | 2019 | papers/fort2019_deep_ensembles_loss_landscape.pdf | Mode diversity; NeurIPS |
| 9 | DICE: Diversity in Deep Ensembles | Ramé, Cord | 2021 | papers/rame2021_DICE_diversity_deep_ensembles.pdf | Conditional redundancy |
| 10 | Diverse Weight Averaging for OOD | Ramé et al. | 2022 | papers/rame2022_diverse_weight_averaging_ood.pdf | DiWA; NeurIPS |
| 11 | Sharpness-Diversity Tradeoff | Lu et al. | 2024 | papers/lu2024_sharpness_diversity_tradeoff.pdf | SharpBalance |
| 12 | Two is Better than One | Jung, Song | 2025 | papers/jung2025_two_better_than_one_ensemble_defense.pdf | Efficient ensemble defense |
| 13 | Ensembles and ECOC | Philippon, Gagné | 2023 | papers/philippon2023_ensembles_ecoc_adversarial.pdf | Error-correcting codes |
| 14 | Diversity Regularization for Robustness | Mehrtens et al. | 2022 | papers/mehrtens2022_diversity_regularization_robustness.pdf | OOD-based diversity reg |
| 15 | Anti-Regularized Ensembles for OOD | de Mathelin et al. | 2023 | papers/demathelin2023_anti_regularized_ensembles_ood.pdf | Anti-regularization |
| 16 | Input-Gradient Particle Inference | Trinh et al. | 2023 | papers/trinh2023_input_gradient_particle_inference.pdf | Particle-based diversity |
| 17 | Situation Monitor: OOD Diversity | Syed et al. | 2024 | papers/syed2024_situation_monitor_ood_diversity.pdf | Zero-shot OOD detection |
| 18 | OOD Multi-Comprehension Ensemble | Xu et al. | 2024 | papers/xu2024_ood_multi_comprehension_ensemble.pdf | Multi-scale OOD |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CIFAR-10 | torchvision | 60K (50K train + 10K test), 340MB | 10-class image classification | datasets/cifar10/ | Primary ID dataset |
| CIFAR-100 | torchvision | 60K (50K train + 10K test), 339MB | 100-class image classification | datasets/cifar100/ | Secondary ID dataset |
| SVHN | torchvision | 26K test, 62MB | 10-class digit recognition | datasets/svhn/ | OOD benchmark dataset |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| RandOpt | github.com/sunrainyg/RandOpt | Neural Thickets official impl | code/RandOpt/ | Core algorithm; LLM-focused |
| ADP | github.com/P2333/Adaptive-Diversity-Promoting | Diversity-promoting ensemble training | code/ADP/ | Key baseline; CIFAR attacks |
| ens-div-ood-detect | github.com/Guoxoug/ens-div-ood-detect | Ensemble diversity for OOD detection | code/ens-div-ood-detect/ | OOD evaluation pipeline |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
- Primary search via arXiv API for "neural thicket" (found core paper)
- Broader search with queries combining: ensemble diversity, adversarial robustness, OOD detection, model soups, weight-space perturbation
- Cross-referenced citations from Neural Thickets paper and related work sections
- Searched GitHub for official implementations

### Selection Criteria
- Papers directly addressing the intersection of ensembles + adversarial/OOD robustness
- Papers from the same research group (Yulu Gan appears in both Neural Thickets and Split-Ensemble)
- Foundational works on ensemble diversity (Fort et al., Pang et al.)
- Recent works (2022-2026) preferred for state-of-the-art context
- Datasets matching those used in the adversarial robustness literature

### Challenges Encountered
- Paper-finder service timed out; used arXiv API directly as fallback
- Neural Thickets is very recent (March 2026) so no follow-up work exists yet
- No existing work directly evaluates thicket ensembles for adversarial/OOD robustness — this is genuinely novel
- RandOpt code is designed for LLMs; will need adaptation for vision models

### Gaps and Workarounds
- No pretrained adversarially-robust models downloaded (can use RobustBench library at experiment time)
- Tiny-ImageNet not downloaded (can be added if needed, but CIFAR-10/100 + SVHN covers the core experiments)
- No LLM models downloaded locally (RandOpt experiments require GPU cluster; vision experiments are more feasible)

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **CIFAR-10** as in-distribution dataset (most comparable with existing adversarial robustness literature)
- **SVHN** as out-of-distribution dataset
- **CIFAR-100** for evaluating how class count affects diversity and robustness

### 2. Baseline Methods
- Single pretrained model (ResNet-18 or ResNet-50)
- Deep Ensemble (5 independently trained models)
- RandOpt/Thicket Ensemble (random Gaussian perturbations around pretrained weights)
- ADP ensemble (diversity-promoting training as comparison)

### 3. Evaluation Metrics
- Clean accuracy, PGD robust accuracy (ℓ∞, ε=8/255)
- AUROC and FPR@95 for OOD detection
- Spectral Discordance for diversity measurement
- Computational cost comparison

### 4. Code to Adapt/Reuse
- **RandOpt** (code/RandOpt/): Adapt perturbation logic (θ' = θ + σ·ε) for vision models
- **ADP** (code/ADP/): Use as baseline; adapt diversity loss for thicket selection
- **ens-div-ood-detect** (code/ens-div-ood-detect/): Reuse OOD evaluation pipeline

### 5. Key Experimental Variables
- Perturbation scale σ (controls neighborhood size)
- Population size N (number of random perturbations)
- Ensemble size K (top-K selection)
- Perturbation strategy: isotropic Gaussian vs. orthogonal vs. adversarial directions
- Selection criterion: task performance vs. diversity-aware selection
