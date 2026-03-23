# Literature Review: Adversarial and OOD Robustness of Neural Thicket Ensembles

## Research Area Overview

This review covers the intersection of three research areas: (1) **Neural Thickets** — the recently discovered phenomenon that pretrained model weights are surrounded by dense, diverse task-specific experts; (2) **Adversarial robustness of ensembles** — how ensemble diversity affects resilience to adversarial attacks; and (3) **Out-of-distribution (OOD) detection with ensembles** — leveraging ensemble disagreement/diversity for detecting distributional shift.

The central research hypothesis is that neural thicket ensembles (formed by randomly sampling weight perturbations around pretrained weights) improve robustness compared to single models, but their vulnerability to adversarial and OOD inputs depends on the diversity and independence of the sampled experts. Explicitly encouraging diversity (e.g., via orthogonal perturbations) should further enhance resilience.

---

## Key Papers

### 1. Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights
- **Authors**: Yulu Gan, Phillip Isola (MIT CSAIL)
- **Year**: 2026
- **Source**: arXiv:2603.12228
- **Key Contribution**: Discovers that large pretrained models are surrounded by a dense "thicket" of task-specific expert solutions in weight space. Random Gaussian perturbations of pretrained weights can find task-improving specialists.
- **Methodology**:
  - **RandOpt algorithm**: Sample N random weight perturbations θ' = θ + σ·ε(s), evaluate each on a training set, select top-K, ensemble predictions via majority vote.
  - Measures **solution density** δ(m) — fraction of random perturbations that improve performance by margin m.
  - Measures **solution diversity** via Spectral Discordance D — correlation structure across tasks shows specialists, not generalists.
- **Datasets Used**: GSM8K, MATH-500, OlympiadBench, Countdown, MBPP, ROCStories, USPTO, GQA
- **Models**: Qwen2.5 (0.5B-32B), Llama3.1-8B, OLMo3-7B
- **Results**:
  - Solution density scales with model size (larger models → denser thickets)
  - Spectral discordance increases with scale (more diverse specialists)
  - RandOpt (K=50) competitive with PPO, GRPO, ES under same training FLOPs
  - O(1) training time (fully parallel)
  - Distillation can compress ensemble to single model
- **Code Available**: https://github.com/sunrainyg/RandOpt
- **Relevance**: THE core paper. Establishes the neural thicket phenomenon. Our research probes whether these thicket ensembles are robust to adversarial/OOD inputs, and whether diversity-encouraging sampling improves robustness.
- **Limitations noted**:
  - Only tested on tasks with categorical/integer answers (majority vote)
  - No adversarial robustness evaluation
  - No OOD detection evaluation
  - Perturbations are isotropic Gaussian — no exploration of structured/directed perturbations

### 2. Improving Adversarial Robustness via Promoting Ensemble Diversity (ADP)
- **Authors**: Tianyu Pang, Kun Xu, Chao Du, Ning Chen, Jun Zhu (Tsinghua)
- **Year**: 2019
- **Source**: ICML 2019, arXiv:1901.08846
- **Key Contribution**: Defines ensemble diversity in the adversarial setting as diversity among *non-maximal* predictions. Proposes Adaptive Diversity Promoting (ADP) regularizer.
- **Methodology**:
  - ADP regularizer = log-ensemble-diversity (LED) + ensemble entropy
  - Encourages non-maximal predictions to be mutually orthogonal across ensemble members
  - Simultaneous training of all members
  - Diversity reduces adversarial transferability between members
- **Datasets Used**: MNIST, CIFAR-10, CIFAR-100
- **Attacks tested**: FGSM, BIM, PGD, MIM, JSMA, C&W, EAD
- **Results**: Significantly improves adversarial robustness while maintaining clean accuracy
- **Code Available**: https://github.com/P2333/Adaptive-Diversity-Promoting
- **Relevance**: Directly relevant — shows that promoting ensemble diversity improves adversarial robustness. Key question: does the *natural* diversity of neural thicket ensembles achieve similar benefits? Can ADP-style regularization be applied to thicket sampling?

### 3. Split-Ensemble: Efficient OOD-aware Ensemble via Task and Model Splitting
- **Authors**: Anthony Chen*, Huanrui Yang*, **Yulu Gan** (same author as Neural Thickets!), et al.
- **Year**: 2024
- **Source**: ICML 2024, arXiv:2312.09148
- **Key Contribution**: Splits classification task into complementary subtasks; each submodel treats other subtasks' data as OOD proxy. Tree-like architecture with shared low-level features.
- **Methodology**:
  - Subtask splitting based on feature similarity
  - OOD-aware training without external OOD data
  - Iterative splitting and pruning for efficient architecture
- **Datasets Used**: CIFAR-10, CIFAR-100, Tiny-ImageNet (ID); SVHN, LSUN, iSUN, Textures, Places365 (OOD)
- **Results**: +0.8%, +1.8%, +25.5% accuracy improvement on CIFAR-10/100/TinyImageNet. OOD detection improves by 2.2-29.6% AUROC over single model.
- **Relevance**: Same first author as Neural Thickets. Shows how ensemble diversity can be engineered for OOD detection. The subtask splitting concept could inspire diversity-encouraging strategies for thicket ensembles.

### 4. Seasoning Model Soups for Robustness to Adversarial and Natural Distribution Shifts
- **Authors**: Francesco Croce, Sylvestre-Alvise Rebuffi, Evan Shelhamer, Sven Gowal (DeepMind)
- **Year**: 2023
- **Source**: CVPR 2023, arXiv:2302.10164
- **Key Contribution**: Shows that linear combinations of adversarially-trained model parameters (model soups) can smoothly trade off robustness to different ℓp-norm adversaries.
- **Methodology**:
  - Pre-train single robust model → fine-tune to different threat models
  - Create soups: θ^w = Σ w_i · θ^i (convex or extrapolated combinations)
  - Adapt to unseen shifts with few examples by tuning soup weights
- **Datasets Used**: CIFAR-10, ImageNet (with ℓ∞, ℓ2, ℓ1 attacks)
- **Results**: Competitive with multi-norm adversarial training; can control robustness level without retraining; adaptation to distribution shifts from few examples
- **Relevance**: Model soups operate in the same weight-space interpolation paradigm as neural thickets. Shows that weight-space ensembling can achieve multi-threat robustness. Key parallel: thicket perturbations are random directions in weight space; model soups use structured directions (fine-tuned models).

### 5. On the Usefulness of Deep Ensemble Diversity for Out-of-Distribution Detection
- **Authors**: Guoxuan Xia, Christos-Savvas Bouganis (Imperial College London)
- **Year**: 2022
- **Source**: UNCV-W 2022, arXiv:2207.07517
- **Key Contribution**: Shows that ensemble diversity (Mutual Information) is NOT reliably useful for OOD detection at ImageNet scale. Proposes averaging task-specific scores (e.g., Energy) over ensemble instead.
- **Methodology**:
  - Decomposes ensemble uncertainty: Total = Average + Diversity (MI)
  - MI performs 30-40% worse than single-model entropy on some OOD datasets
  - Alternative: average Energy score over ensemble members
- **Datasets Used**: ImageNet-200 (ID); Near-ImageNet-200, Caltech-45, Openimage-O, iNaturalist, Textures, Colorectal, Colonoscopy, ImageNet-O (OOD)
- **Code Available**: https://github.com/Guoxoug/ens-div-ood-detect
- **Relevance**: Critical counterpoint — ensemble diversity alone may not help OOD detection. For thicket ensembles, the *type* of diversity matters. MI fails because average uncertainty already separates ID/OOD well. Exception: MI works on adversarially-constructed ImageNet-O, suggesting diversity IS useful against adversarial-style OOD.

### 6. Ensemble Adversarial Defense via Multiple Dispersed Low Curvature Models
- **Authors**: Kaikang Zhao, Xi Chen, Wei Huang
- **Year**: 2024
- **Source**: arXiv:2403.16405
- **Key Contribution**: Integration of diverse low-curvature models for ensemble adversarial defense. Diversity among sub-models increases attack cost.
- **Relevance**: Shows that diversity + flatness of loss landscape improves ensemble robustness. Thickets exist in flat basins — connection to this work.

### 7. MEAT: Median-Ensemble Adversarial Training
- **Authors**: Zhaozhe Hu, Jia-Li Yin, Bin Chen
- **Year**: 2024
- **Source**: arXiv:2406.14259
- **Key Contribution**: Uses median (rather than mean) for weight averaging in self-ensemble adversarial training, improving robustness and generalization.
- **Relevance**: Alternative ensembling strategies beyond majority vote — applicable to how thicket experts are combined.

### 8. Deep Ensembles: A Loss Landscape Perspective
- **Authors**: Stanislav Fort, Huiyi Hu, Balaji Lakshminarayanan
- **Year**: 2019
- **Source**: NeurIPS 2019, arXiv:1912.02757
- **Key Contribution**: Shows deep ensembles explore different modes of the loss landscape; functional diversity arises from mode diversity.
- **Relevance**: Theoretical foundation — thicket perturbations may or may not access different modes. If perturbations stay within one mode, diversity may be limited.

### 9. DICE: Diversity in Deep Ensembles via Conditional Redundancy Adversarial Estimation
- **Authors**: Alexandre Ramé, Matthieu Cord
- **Year**: 2021
- **Source**: arXiv:2101.05544
- **Key Contribution**: Regularizes ensemble predictions to increase diversity while maintaining accuracy using adversarial estimation of conditional redundancy.
- **Relevance**: Another diversity-promoting method for ensembles — could be applied to thicket expert selection.

### 10. Diverse Weight Averaging for Out-of-Distribution Generalization
- **Authors**: Alexandre Ramé et al.
- **Year**: 2022
- **Source**: NeurIPS 2022, arXiv:2205.09739
- **Key Contribution**: Averages weights of diverse models for OOD generalization. Shows weight-space diversity translates to functional diversity.
- **Relevance**: Weight averaging (like thicket ensembles) can improve OOD generalization.

### 11. Sharpness-diversity tradeoff: SharpBalance
- **Authors**: Haiquan Lu et al.
- **Year**: 2024
- **Source**: arXiv:2407.12996
- **Key Contribution**: Identifies tradeoff between sharpness of individual learners and diversity of ensemble. Proposes SharpBalance to navigate this.
- **Relevance**: Thicket perturbation scale σ may affect this tradeoff — larger σ → more diversity but potentially sharper minima.

---

## Common Methodologies

### Ensemble Construction
- **Independent training**: Train members separately (Deep Ensembles)
- **Simultaneous training with diversity regularization**: ADP, DICE
- **Weight-space perturbation**: RandOpt/Neural Thickets, Bayesian NNs
- **Weight-space interpolation**: Model Soups, Diverse Weight Averaging
- **Architecture splitting**: Split-Ensemble

### Adversarial Attack Methods
- White-box: PGD, AutoPGD, C&W
- Black-box/transfer: FGSM, MIM
- Multi-norm: ℓ∞, ℓ2, ℓ1 bounded

### OOD Detection Metrics
- AUROC, FPR@95 (standard binary classification metrics)
- Uncertainty scores: Entropy, Energy, Maximum Softmax Probability (MSP), Mutual Information

---

## Standard Baselines
- **Single model** (no ensemble)
- **Deep Ensemble** (independently trained members, averaged predictions)
- **Adversarial training** (PGD-AT) for robustness baselines
- **Model Soups** for weight-space interpolation baselines
- **ADP** for diversity-promoting ensemble baseline

## Evaluation Metrics
- **Clean accuracy**: Standard test accuracy without perturbation
- **Robust accuracy**: Accuracy under adversarial attack (PGD, AutoPGD)
- **AUROC/FPR@95**: OOD detection performance
- **Spectral Discordance**: Diversity measure from Neural Thickets paper
- **Mutual Information**: Ensemble disagreement measure

## Datasets in the Literature
- **CIFAR-10**: Used by ADP, Split-Ensemble, Model Soups, SharpBalance (most common)
- **CIFAR-100**: Used by ADP, Split-Ensemble, diversity regularization papers
- **ImageNet/variants**: Used by Model Soups, Xia et al., for large-scale evaluation
- **SVHN**: Common OOD dataset when CIFAR is ID
- **GSM8K, MATH-500, Countdown**: Used by Neural Thickets (LLM reasoning tasks)

---

## Gaps and Opportunities

1. **No adversarial evaluation of neural thickets**: The Neural Thickets paper evaluates accuracy but never tests robustness to adversarial attacks. This is the primary gap.

2. **No OOD evaluation of neural thickets**: Similarly, no evaluation of whether thicket ensembles can detect OOD inputs.

3. **Isotropic vs. structured perturbations**: Neural Thickets uses isotropic Gaussian noise. The hypothesis that orthogonal/adversarial perturbation directions improve diversity is untested.

4. **Vision domain gap**: Neural Thickets focuses on LLMs; extending to vision models (where adversarial robustness is better studied) would enable direct comparison with ADP, Model Soups, etc.

5. **Diversity-robustness relationship**: The relationship between spectral discordance (diversity metric from Neural Thickets) and adversarial/OOD robustness is unexplored.

6. **Selection strategies**: Top-K selection in RandOpt is based on task performance; selecting for diversity or robustness could yield different ensemble properties.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **CIFAR-10** (primary): Most well-studied for adversarial robustness; enables direct comparison with ADP, Model Soups baselines
2. **CIFAR-100** (secondary): More classes → richer diversity structure
3. **SVHN** (OOD benchmark): Standard OOD dataset when CIFAR is ID

### Recommended Baselines
1. **Single pretrained model** (no perturbation)
2. **Deep Ensemble** (independently trained from different initializations)
3. **RandOpt** (random Gaussian perturbations, top-K, majority vote) — the vanilla thicket ensemble
4. **ADP ensemble** (diversity-promoting training)
5. **Adversarial training** (PGD-AT) single model

### Recommended Metrics
1. Clean accuracy
2. Robust accuracy under PGD attack (ℓ∞, ε=8/255 for CIFAR)
3. AUROC and FPR@95 for OOD detection (CIFAR→SVHN)
4. Spectral Discordance (diversity measure)
5. Ensemble disagreement / Mutual Information

### Methodological Considerations
- Use pretrained vision models (e.g., ResNet-18/50 pretrained on CIFAR or ImageNet) as the base for thicket sampling
- Compare isotropic Gaussian perturbations vs. orthogonal perturbations vs. adversarially-selected perturbation directions
- Measure how perturbation scale σ affects the diversity-robustness tradeoff
- Test both white-box (all ensemble members known) and transfer attacks
- For OOD, test both diversity-based scores (MI) and averaging-based scores (mean Energy)
