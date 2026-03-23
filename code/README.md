# Cloned Repositories

## Repo 1: RandOpt (Neural Thickets)
- **URL**: https://github.com/sunrainyg/RandOpt
- **Purpose**: Official implementation of the Neural Thickets paper (RandOpt algorithm)
- **Location**: code/RandOpt/
- **Key files**:
  - `randopt.py` — Main RandOpt implementation
  - `core/` — Core evaluation and perturbation logic
  - `baselines/` — PPO, GRPO, ES baseline implementations
  - `distillation/` — Distillation of ensemble to single model
  - `simple_1D_signals_expts/` — Minimal thicket demonstration
  - `scripts/` — Slurm and local run scripts
  - `data/` — Dataset loading utilities
- **Dependencies**: See `requirements.txt` (vLLM, transformers, torch)
- **Notes**:
  - Designed for LLM post-training (requires GPU + vLLM)
  - The perturbation logic (θ' = θ + σ·ε) can be adapted for vision models
  - Key hyperparameters: N (population size), K (ensemble size), σ (noise scales)
  - The `simple_1D_signals_expts/` provides a minimal, self-contained demo

## Repo 2: Adaptive-Diversity-Promoting (ADP)
- **URL**: https://github.com/P2333/Adaptive-Diversity-Promoting
- **Purpose**: ADP regularizer for adversarial robustness via ensemble diversity
- **Location**: code/ADP/
- **Key files**:
  - Training scripts for MNIST, CIFAR-10, CIFAR-100
  - ADP loss implementation (LED + entropy terms)
  - Attack evaluation code
- **Notes**:
  - Key baseline for comparing diversity-promoting approaches
  - The ADP loss can be adapted to guide thicket ensemble selection
  - Tests FGSM, BIM, PGD, MIM, C&W, EAD attacks

## Repo 3: ens-div-ood-detect
- **URL**: https://github.com/Guoxoug/ens-div-ood-detect
- **Purpose**: Code for "On the Usefulness of Deep Ensemble Diversity for OOD Detection"
- **Location**: code/ens-div-ood-detect/
- **Key files**:
  - OOD detection evaluation pipeline
  - Ensemble uncertainty decomposition (MI, Ens.H, Av.H)
  - Energy score averaging implementation
- **Notes**:
  - Provides ready-to-use OOD evaluation metrics
  - Can be adapted to evaluate thicket ensembles
  - Uses ImageNet-200 as ID dataset with multiple OOD datasets
