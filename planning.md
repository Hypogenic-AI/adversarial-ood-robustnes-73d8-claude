# Research Plan: Probing the Limits — Adversarial and OOD Robustness of Neural Thicket Ensembles

## Motivation & Novelty Assessment

### Why This Research Matters
Neural thickets (Gan & Isola, 2026) reveal that pretrained models are surrounded by dense clusters of task-specific expert solutions reachable via simple random perturbations. If these ensembles are robust to adversarial attacks and out-of-distribution (OOD) inputs, they offer a remarkably cheap alternative to adversarial training or expensive deep ensembles. Understanding their failure modes is critical before deployment.

### Gap in Existing Work
The Neural Thickets paper evaluates only clean accuracy on LLM reasoning tasks. No prior work has:
1. Tested thicket ensembles against adversarial attacks (PGD, FGSM)
2. Evaluated thicket ensembles for OOD detection
3. Extended the thicket concept to vision models (where adversarial robustness is better studied)
4. Compared isotropic vs. structured perturbation strategies for ensemble diversity

### Our Novel Contribution
We transplant the neural thicket ensemble concept to vision (ResNet on CIFAR-10), then systematically evaluate:
- Adversarial robustness under white-box PGD and FGSM attacks
- OOD detection (CIFAR-10 → SVHN) using ensemble uncertainty
- Whether diversity-encouraging perturbation strategies (orthogonal, adversarial directions) improve robustness beyond random Gaussian perturbations

### Experiment Justification
- **Exp 1 (Baseline training)**: Need a well-trained single model as the "center" of the thicket
- **Exp 2 (Thicket construction)**: Test whether random weight perturbations yield diverse, functional ensemble members in the vision domain
- **Exp 3 (Adversarial robustness)**: Core test — does the thicket ensemble resist PGD/FGSM better than a single model?
- **Exp 4 (Perturbation strategies)**: Test the hypothesis that orthogonal/structured perturbations improve diversity and thus robustness
- **Exp 5 (OOD detection)**: Test whether thicket ensemble disagreement signals OOD inputs
- **Exp 6 (Ablations)**: How do ensemble size K and perturbation scale σ affect results?

## Research Question
Does the natural diversity of neural thicket ensembles (random weight perturbations around a pretrained model) confer adversarial and OOD robustness in vision tasks? Can structured perturbation strategies further improve resilience?

## Hypothesis Decomposition
1. **H1**: Thicket ensembles achieve higher adversarial robustness than a single model (majority vote filters out per-member adversarial errors)
2. **H2**: Thicket ensembles with orthogonal perturbations are more robust than those with isotropic Gaussian perturbations (greater functional diversity)
3. **H3**: Thicket ensemble disagreement (mutual information) is a useful OOD detector
4. **H4**: There exists an optimal perturbation scale σ that trades off clean accuracy vs. robustness

## Proposed Methodology

### Approach
Adapt the RandOpt/thicket concept to vision: start with a pretrained ResNet-18 on CIFAR-10, generate N=100 weight perturbations at various scales, select top-K=10 by validation accuracy, ensemble predictions via majority vote. Compare perturbation strategies and evaluate under attack.

### Experimental Steps
1. Train a ResNet-18 on CIFAR-10 to ~93%+ accuracy (or use a pretrained checkpoint)
2. Generate perturbed models: θ' = θ + σ·ε where ε is drawn from different distributions
   - Isotropic Gaussian (baseline thicket)
   - Orthogonal perturbations (Gram-Schmidt on random directions)
   - Adversarial perturbation directions (maximize loss diversity)
3. Select top-K by validation accuracy → form ensembles
4. Evaluate clean accuracy of all ensembles
5. Attack with PGD (ℓ∞, ε=8/255) and FGSM — measure robust accuracy
6. OOD detection: compute ensemble uncertainty on CIFAR-10 test vs. SVHN, measure AUROC
7. Ablate over K ∈ {3, 5, 10} and σ ∈ {0.001, 0.005, 0.01, 0.05}

### Baselines
1. **Single model** (no perturbation) — lower bound
2. **Random Gaussian thicket ensemble** — vanilla thicket
3. **Deep ensemble** (3 independently trained ResNet-18s) — upper bound for diversity
4. **Adversarial training** (PGD-AT) single model — gold standard for robustness

### Evaluation Metrics
- **Clean accuracy**: Standard test accuracy
- **Robust accuracy**: Accuracy under PGD-20 (ε=8/255, step=2/255) and FGSM
- **AUROC**: OOD detection (CIFAR-10 vs SVHN) using max softmax prob, energy score, and mutual information
- **FPR@95**: False positive rate at 95% true positive rate for OOD
- **Ensemble disagreement**: Average pairwise disagreement rate among members

### Statistical Analysis Plan
- Report mean ± std over 3 random seeds for all metrics
- Paired t-tests comparing thicket ensemble vs. single model robustness
- Effect size (Cohen's d) for robustness improvements
- Significance level α = 0.05

## Expected Outcomes
- H1 supported: Thicket ensembles should improve robust accuracy by 2-10% over single model
- H2 partially supported: Orthogonal perturbations may help but effect may be small if thicket diversity is already high
- H3 mixed: Based on Xia et al., MI may not help; averaged energy scores likely better
- H4 supported: Very small σ → too similar members; very large σ → members too degraded

## Timeline and Milestones
1. Environment setup + base model training: ~15 min
2. Thicket construction + perturbation strategies: ~20 min
3. Adversarial evaluation: ~30 min
4. OOD evaluation: ~15 min
5. Ablations: ~20 min
6. Analysis + documentation: ~30 min

## Potential Challenges
- Weight perturbations may completely destroy model performance in vision (unlike LLMs which are over-parameterized)
- Need to find the right σ scale for ResNet-18 on CIFAR-10
- PGD attack on ensemble is computationally expensive
- If perturbations don't yield functional models, will reduce σ or perturb only specific layers

## Success Criteria
- Complete evaluation of thicket ensembles under adversarial + OOD conditions
- Clear comparison showing whether diversity helps or not
- Identification of optimal perturbation strategy and scale
- Actionable recommendations for practitioners
