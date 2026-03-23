"""Generate all visualizations for the research report."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / "results.json") as f:
    R = json.load(f)

plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# ─── Figure 1: Sigma sweep — accuracy vs diversity tradeoff ──────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))
sigmas = [0.001, 0.005, 0.01, 0.02, 0.05]
clean_accs = [R['sigma_sweep'][str(s)]['clean_acc_vote'] for s in sigmas]
disagreements = [R['sigma_sweep'][str(s)]['disagreement'] for s in sigmas]

color1, color2 = '#2196F3', '#FF5722'
ax1.set_xlabel('Perturbation Scale σ')
ax1.set_ylabel('Clean Accuracy (%)', color=color1)
ax1.plot(sigmas, clean_accs, 'o-', color=color1, linewidth=2, markersize=8, label='Clean Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
ax2.set_ylabel('Disagreement Rate', color=color2)
ax2.plot(sigmas, disagreements, 's--', color=color2, linewidth=2, markersize=8, label='Disagreement')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Neural Thicket: Accuracy vs. Diversity Tradeoff', fontsize=14, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'sigma_sweep.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Figure 2: Adversarial robustness comparison ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['Single\nModel', 'Gaussian\nThicket', 'Orthogonal\nThicket', 'Layer-Scaled\nThicket', 'Deep\nEnsemble', 'Adversarial\nTraining']
clean = [R['base_model']['clean_acc'], R['gaussian_thicket']['clean_acc_vote'],
         R['orthogonal_thicket']['clean_acc_vote'], R['layer_scaled_thicket']['clean_acc_vote'],
         R['deep_ensemble']['clean_acc_vote'], R['at_model']['clean_acc']]
pgd = [R['base_model']['pgd_acc'], R['gaussian_thicket']['pgd_acc_vote'],
       R['orthogonal_thicket']['pgd_acc_vote'], R['layer_scaled_thicket']['pgd_acc_vote'],
       R['deep_ensemble']['pgd_acc_vote'], R['at_model']['pgd_acc']]
fgsm = [R['base_model']['fgsm_acc'], R['gaussian_thicket']['fgsm_acc'],
        R['orthogonal_thicket']['fgsm_acc'], R['layer_scaled_thicket']['fgsm_acc'],
        R['deep_ensemble']['fgsm_acc'], R['at_model']['fgsm_acc']]

x = np.arange(len(methods))
width = 0.25
bars1 = ax.bar(x - width, clean, width, label='Clean', color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x, pgd, width, label='PGD-20', color='#F44336', alpha=0.85)
bars3 = ax.bar(x + width, fgsm, width, label='FGSM', color='#FF9800', alpha=0.85)

ax.set_ylabel('Accuracy (%)')
ax.set_title('Adversarial Robustness: Thicket Ensembles vs. Baselines', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 3:
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'adversarial_robustness.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Figure 3: OOD Detection Comparison ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ood_methods = ['Gaussian\nThicket', 'Orthogonal\nThicket', 'Layer-Scaled\nThicket', 'Deep\nEnsemble']
ood_keys = ['gaussian_thicket', 'orthogonal_thicket', 'layer_scaled_thicket', 'deep_ensemble']
metrics = ['msp', 'energy', 'mutual_info']
metric_labels = ['MSP', 'Energy', 'Mutual Info']
colors = ['#2196F3', '#4CAF50', '#9C27B0']

# AUROC
x = np.arange(len(ood_methods))
width = 0.22
for i, (m, label, c) in enumerate(zip(metrics, metric_labels, colors)):
    vals = [R['ood_detection'][k][m]['auroc'] for k in ood_keys]
    ax1.bar(x + (i - 1) * width, vals, width, label=label, color=c, alpha=0.85)
ax1.set_ylabel('AUROC')
ax1.set_title('OOD Detection: AUROC (CIFAR-10 → SVHN)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(ood_methods, fontsize=9)
ax1.legend()
ax1.set_ylim(0.85, 0.95)
ax1.grid(axis='y', alpha=0.3)

# FPR@95
for i, (m, label, c) in enumerate(zip(metrics, metric_labels, colors)):
    vals = [R['ood_detection'][k][m]['fpr95'] for k in ood_keys]
    ax2.bar(x + (i - 1) * width, vals, width, label=label, color=c, alpha=0.85)
ax2.set_ylabel('FPR@95')
ax2.set_title('OOD Detection: FPR@95 (lower is better)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(ood_methods, fontsize=9)
ax2.legend()
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ood_detection.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Figure 4: K ablation ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ks = [3, 5, 10]
clean_k = [R['k_ablation'][str(k)]['clean_acc_vote'] for k in ks]
pgd_k = [R['k_ablation'][str(k)]['pgd_acc_vote'] for k in ks]

ax.plot(ks, clean_k, 'o-', color='#4CAF50', linewidth=2, markersize=10, label='Clean Accuracy')
ax.plot(ks, pgd_k, 's--', color='#F44336', linewidth=2, markersize=10, label='PGD-20 Accuracy')
ax.set_xlabel('Ensemble Size K')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Effect of Ensemble Size K (σ=0.001)', fontsize=14, fontweight='bold')
ax.set_xticks(ks)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'k_ablation.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Figure 5: Summary heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
methods_hm = ['Single Model', 'Gaussian Thicket', 'Ortho Thicket', 'LayerScale Thicket', 'Deep Ensemble', 'Adv. Training']
metrics_hm = ['Clean Acc', 'PGD Acc', 'FGSM Acc', 'OOD AUROC\n(Energy)', 'OOD FPR@95\n(Energy)']

data = np.array([
    [R['base_model']['clean_acc'], R['base_model']['pgd_acc'], R['base_model']['fgsm_acc'], 0, 0],
    [R['gaussian_thicket']['clean_acc_vote'], R['gaussian_thicket']['pgd_acc_vote'], R['gaussian_thicket']['fgsm_acc'],
     R['ood_detection']['gaussian_thicket']['energy']['auroc']*100, (1-R['ood_detection']['gaussian_thicket']['energy']['fpr95'])*100],
    [R['orthogonal_thicket']['clean_acc_vote'], R['orthogonal_thicket']['pgd_acc_vote'], R['orthogonal_thicket']['fgsm_acc'],
     R['ood_detection']['orthogonal_thicket']['energy']['auroc']*100, (1-R['ood_detection']['orthogonal_thicket']['energy']['fpr95'])*100],
    [R['layer_scaled_thicket']['clean_acc_vote'], R['layer_scaled_thicket']['pgd_acc_vote'], R['layer_scaled_thicket']['fgsm_acc'],
     R['ood_detection']['layer_scaled_thicket']['energy']['auroc']*100, (1-R['ood_detection']['layer_scaled_thicket']['energy']['fpr95'])*100],
    [R['deep_ensemble']['clean_acc_vote'], R['deep_ensemble']['pgd_acc_vote'], R['deep_ensemble']['fgsm_acc'],
     R['ood_detection']['deep_ensemble']['energy']['auroc']*100, (1-R['ood_detection']['deep_ensemble']['energy']['fpr95'])*100],
    [R['at_model']['clean_acc'], R['at_model']['pgd_acc'], R['at_model']['fgsm_acc'], 0, 0],
])

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(np.arange(len(metrics_hm)))
ax.set_yticks(np.arange(len(methods_hm)))
ax.set_xticklabels(metrics_hm)
ax.set_yticklabels(methods_hm)

for i in range(len(methods_hm)):
    for j in range(len(metrics_hm)):
        val = data[i, j]
        if val > 0:
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=10,
                   color='white' if val < 40 else 'black')
        else:
            ax.text(j, i, 'N/A', ha='center', va='center', fontsize=9, color='gray')

ax.set_title('Summary: All Methods × All Metrics', fontsize=14, fontweight='bold')
plt.colorbar(im, label='Score (%)', shrink=0.8)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'summary_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("All visualizations saved to", PLOTS_DIR)
print("Files:", [f.name for f in PLOTS_DIR.glob('*.png')])
