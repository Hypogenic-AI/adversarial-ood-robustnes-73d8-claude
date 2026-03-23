# Probing the Limits: Adversarial and OOD Robustness of Neural Thicket Ensembles

## Overview
This project tests whether neural thicket ensembles — formed by random weight perturbations around a pretrained vision model (ResNet-18 on CIFAR-10) — are robust to adversarial attacks (PGD, FGSM) and can detect out-of-distribution inputs (SVHN). We compare isotropic Gaussian, orthogonal, and layer-scaled perturbation strategies against deep ensembles and adversarial training baselines.

## Key Findings

- **Thicket ensembles provide zero adversarial robustness** in vision: PGD accuracy is 12.5% (identical to the single model's 12.6%). Adversarial training achieves 78.5%.
- **Vision models have catastrophically tight loss basins**: Perturbations of σ≥0.02 destroy ResNet-18 entirely (94% → 28% accuracy). Viable perturbation scales (σ≤0.001) produce near-clone ensemble members with <1% disagreement.
- **Structured perturbations don't help**: Orthogonal and layer-scaled perturbations perform identically to isotropic Gaussian.
- **OOD detection is comparable across methods**: All ensembles achieve ~0.93 AUROC using energy scores (CIFAR-10 → SVHN). Diversity-based metrics (MI) underperform.
- **Deep ensembles offer modest improvement**: 19.7% PGD accuracy (vs 12.6% single model) from true functional diversity, but still far from adversarial training.

## How to Reproduce

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy matplotlib scipy scikit-learn tqdm

# Run experiments (requires GPU, ~90 min)
python src/experiment.py

# Generate visualizations
python src/visualize.py
```

## File Structure
```
├── REPORT.md              # Full research report with results
├── planning.md            # Experimental design and motivation
├── src/
│   ├── experiment.py      # Main experiment pipeline
│   └── visualize.py       # Visualization generation
├── results/
│   ├── results.json       # All numerical results
│   ├── models/            # Trained model checkpoints
│   └── plots/             # Generated figures
├── datasets/              # CIFAR-10, CIFAR-100, SVHN
├── papers/                # 18 related research papers (PDFs)
├── code/                  # Reference implementations (RandOpt, ADP, ens-div-ood-detect)
├── literature_review.md   # Comprehensive literature review
└── resources.md           # Resource catalog
```

See [REPORT.md](REPORT.md) for the full research report.
