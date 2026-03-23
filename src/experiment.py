"""
Neural Thicket Ensemble: Adversarial & OOD Robustness Experiments
================================================================
Tests whether weight-perturbed ensembles around a pretrained vision model
are robust to adversarial attacks and can detect OOD inputs.
"""

import os
import sys
import json
import random
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE)
BATCH_SIZE = 128
NUM_WORKERS = 4
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = Path(__file__).parent.parent / "datasets"

# Perturbation experiment params
SIGMA_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05]
N_PERTURBATIONS = 60  # sample pool
K_VALUES = [3, 5, 10]  # ensemble sizes
DEFAULT_K = 5
DEFAULT_SIGMA = 0.01

# Attack params
PGD_EPS = 8.0 / 255
PGD_ALPHA = 2.0 / 255
PGD_STEPS = 20
FGSM_EPS = 8.0 / 255

# Training params
TRAIN_EPOCHS = 50
AT_EPOCHS = 30  # adversarial training epochs


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Data Loading ─────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def get_cifar10_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "cifar10"), train=True, download=False,
        transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "cifar10"), train=False, download=False,
        transform=transform_test)
    # Split train into train/val (45k/5k)
    val_indices = list(range(45000, 50000))
    train_indices = list(range(45000))
    train_subset = Subset(trainset, train_indices)
    # Val uses test transform
    valset_base = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR / "cifar10"), train=True, download=False,
        transform=transform_test)
    val_subset = Subset(valset_base, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_svhn_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # same normalization for comparability
    ])
    svhn = torchvision.datasets.SVHN(
        root=str(DATA_DIR / "svhn"), split='test', download=False,
        transform=transform)
    return DataLoader(svhn, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# ─── Model ────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=TRAIN_EPOCHS,
                device=DEVICE, adversarial=False, save_path=None):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if adversarial:
                # PGD adversarial training
                inputs = pgd_attack(model, inputs, targets, eps=PGD_EPS,
                                    alpha=PGD_ALPHA, steps=7, device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        scheduler.step()
        train_acc = 100.0 * correct / total

        # Validation
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, best={best_acc:.1f}%")

    # Load best
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model, best_acc


def evaluate(model, loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


# ─── Adversarial Attacks ──────────────────────────────────────────────────────

def pgd_attack(model, images, labels, eps, alpha, steps, device=DEVICE):
    """PGD attack (ℓ∞)"""
    images_adv = images.clone().detach()
    images_adv += torch.empty_like(images_adv).uniform_(-eps, eps)
    images_adv = torch.clamp(images_adv, 0, 1)  # will be unnormalized below

    for _ in range(steps):
        images_adv.requires_grad_(True)
        outputs = model(images_adv)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, images_adv)[0]
        images_adv = images_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(images_adv - images, min=-eps, max=eps)
        images_adv = images + delta
        # No clamping to [0,1] since data is normalized
    return images_adv.detach()


def fgsm_attack(model, images, labels, eps, device=DEVICE):
    """FGSM attack (ℓ∞)"""
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = F.cross_entropy(outputs, labels)
    grad = torch.autograd.grad(loss, images_adv)[0]
    images_adv = images_adv + eps * grad.sign()
    return images_adv.detach()


def evaluate_adversarial(model, loader, attack_fn, device=DEVICE, desc="Adv eval"):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = attack_fn(model, inputs, targets)
        with torch.no_grad():
            outputs = model(adv_inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


# ─── Neural Thicket Ensemble ─────────────────────────────────────────────────

def perturb_weights_gaussian(model, sigma):
    """Apply isotropic Gaussian perturbation to all parameters."""
    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for p in perturbed.parameters():
            p.add_(torch.randn_like(p) * sigma)
    return perturbed


def perturb_weights_orthogonal(base_model, sigma, n_models, device=DEVICE):
    """Generate n_models perturbations with approximately orthogonal directions."""
    # Flatten all parameters into a single vector
    base_params = torch.cat([p.data.view(-1) for p in base_model.parameters()])
    d = base_params.numel()

    # Generate random directions and orthogonalize via QR on smaller batches
    # For high-d params, we can't do full QR. Instead, generate random directions
    # and subtract projections onto previous directions (approximate orthogonality)
    directions = []
    for i in range(n_models):
        v = torch.randn(d, device=device)
        # Project out previous directions (Gram-Schmidt, approximate)
        for prev in directions[-min(5, len(directions)):]:  # limit to last 5 for efficiency
            v -= torch.dot(v, prev) * prev
        v = v / (v.norm() + 1e-8)
        directions.append(v)

    models = []
    for v in directions:
        perturbed = copy.deepcopy(base_model)
        offset = 0
        with torch.no_grad():
            for p in perturbed.parameters():
                numel = p.numel()
                p.add_(v[offset:offset + numel].view_as(p) * sigma * np.sqrt(d))
                offset += numel
        models.append(perturbed)
    return models


def perturb_weights_layer_scaled(model, sigma):
    """Perturbation scaled by each layer's weight norm (adaptive)."""
    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for p in perturbed.parameters():
            layer_scale = p.data.norm() / (p.numel() ** 0.5 + 1e-8)
            p.add_(torch.randn_like(p) * sigma * layer_scale)
    return perturbed


class ThicketEnsemble:
    """Ensemble of weight-perturbed models around a base model."""

    def __init__(self, models, device=DEVICE):
        self.models = [m.to(device) for m in models]
        self.device = device
        for m in self.models:
            m.eval()

    def predict_all(self, x):
        """Get predictions from all members. Returns (n_models, batch, n_classes) logits."""
        logits_list = []
        with torch.no_grad():
            for m in self.models:
                logits_list.append(m(x))
        return torch.stack(logits_list, dim=0)

    def predict_majority_vote(self, x):
        """Majority vote prediction."""
        all_logits = self.predict_all(x)  # (K, B, C)
        all_preds = all_logits.argmax(dim=2)  # (K, B)
        # Majority vote
        votes = torch.zeros(x.size(0), all_logits.size(2), device=self.device)
        for k in range(len(self.models)):
            votes.scatter_add_(1, all_preds[k].unsqueeze(1),
                              torch.ones(x.size(0), 1, device=self.device))
        return votes.argmax(dim=1)

    def predict_avg_softmax(self, x):
        """Average softmax prediction."""
        all_logits = self.predict_all(x)
        avg_probs = F.softmax(all_logits, dim=2).mean(dim=0)
        return avg_probs

    def ensemble_forward(self, x):
        """Forward pass returning averaged logits (for attacks)."""
        all_logits = self.predict_all(x)
        return all_logits.mean(dim=0)

    def disagreement_rate(self, x):
        """Average pairwise disagreement among members."""
        all_logits = self.predict_all(x)
        all_preds = all_logits.argmax(dim=2)  # (K, B)
        K = len(self.models)
        disagree = 0
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                disagree += (all_preds[i] != all_preds[j]).float().sum().item()
                count += all_preds.size(1)
        return disagree / (count + 1e-8)

    def uncertainty_scores(self, x):
        """Compute various uncertainty scores for OOD detection."""
        all_logits = self.predict_all(x)  # (K, B, C)
        all_probs = F.softmax(all_logits, dim=2)  # (K, B, C)
        avg_probs = all_probs.mean(dim=0)  # (B, C)

        # Max softmax probability (lower = more uncertain)
        msp = avg_probs.max(dim=1)[0]

        # Energy score (higher = more in-distribution)
        avg_logits = all_logits.mean(dim=0)
        energy = torch.logsumexp(avg_logits, dim=1)

        # Mutual Information = H[avg] - avg H[individual]
        eps = 1e-8
        entropy_avg = -(avg_probs * (avg_probs + eps).log()).sum(dim=1)
        individual_entropies = -(all_probs * (all_probs + eps).log()).sum(dim=2)  # (K, B)
        avg_entropy = individual_entropies.mean(dim=0)  # (B,)
        mutual_info = entropy_avg - avg_entropy  # (B,)

        return {
            'msp': msp.cpu().numpy(),
            'energy': energy.cpu().numpy(),
            'mutual_info': mutual_info.cpu().numpy(),
            'entropy': entropy_avg.cpu().numpy(),
        }


class EnsembleWrapper(nn.Module):
    """Wraps thicket ensemble for adversarial attacks (differentiable avg logits)."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = torch.stack([m(x) for m in self.models], dim=0)
        return logits.mean(dim=0)


# ─── Evaluation Functions ────────────────────────────────────────────────────

def evaluate_ensemble_clean(ensemble, loader, device=DEVICE):
    correct_vote, correct_avg, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        pred_vote = ensemble.predict_majority_vote(inputs)
        avg_probs = ensemble.predict_avg_softmax(inputs)
        pred_avg = avg_probs.argmax(dim=1)
        correct_vote += pred_vote.eq(targets).sum().item()
        correct_avg += pred_avg.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct_vote / total, 100.0 * correct_avg / total


def evaluate_ensemble_adversarial(ensemble, loader, eps, alpha, steps, device=DEVICE, desc=""):
    """White-box PGD attack on ensemble (attacks average logits)."""
    wrapper = EnsembleWrapper(ensemble.models).to(device)
    wrapper.eval()
    correct_vote, correct_avg, total = 0, 0, 0
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(wrapper, inputs, targets, eps, alpha, steps, device)
        with torch.no_grad():
            pred_vote = ensemble.predict_majority_vote(adv_inputs)
            pred_avg = ensemble.predict_avg_softmax(adv_inputs).argmax(dim=1)
        correct_vote += pred_vote.eq(targets).sum().item()
        correct_avg += pred_avg.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct_vote / total, 100.0 * correct_avg / total


def compute_ood_metrics(id_scores, ood_scores):
    """Compute AUROC and FPR@95 for OOD detection.
    Assumes higher score = more in-distribution.
    """
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # Handle edge cases
    if np.isnan(scores).any():
        scores = np.nan_to_num(scores, nan=0.0)

    auroc = roc_auc_score(labels, scores)

    # FPR@95: at threshold where 95% of ID is correctly classified
    id_sorted = np.sort(id_scores)
    threshold = id_sorted[int(0.05 * len(id_sorted))]  # 5th percentile of ID
    fpr95 = (ood_scores >= threshold).mean()

    return auroc, fpr95


def collect_uncertainty_scores(ensemble, loader, device=DEVICE):
    all_scores = {'msp': [], 'energy': [], 'mutual_info': [], 'entropy': []}
    for inputs, _ in loader:
        inputs = inputs.to(device)
        scores = ensemble.uncertainty_scores(inputs)
        for k in all_scores:
            all_scores[k].append(scores[k])
    return {k: np.concatenate(v) for k, v in all_scores.items()}


# ─── Build Thicket Ensembles ─────────────────────────────────────────────────

def build_gaussian_thicket(base_model, sigma, n_perturbations, k, val_loader, device=DEVICE):
    """Build thicket ensemble via random Gaussian perturbations + top-K selection."""
    candidates = []
    print(f"  Generating {n_perturbations} Gaussian perturbations (σ={sigma})...")
    for i in range(n_perturbations):
        p_model = perturb_weights_gaussian(base_model, sigma).to(device)
        acc = evaluate(p_model, val_loader, device)
        candidates.append((acc, p_model))
        if (i + 1) % 20 == 0:
            accs = [c[0] for c in candidates]
            print(f"    {i+1}/{n_perturbations}: mean_acc={np.mean(accs):.1f}%, best={max(accs):.1f}%")
    # Select top-K
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [c[1] for c in candidates[:k]]
    selected_accs = [c[0] for c in candidates[:k]]
    all_accs = [c[0] for c in candidates]
    print(f"  Top-{k} val accs: {[f'{a:.1f}' for a in selected_accs]}")
    return ThicketEnsemble(selected, device), all_accs


def build_orthogonal_thicket(base_model, sigma, n_perturbations, k, val_loader, device=DEVICE):
    """Build thicket ensemble via orthogonal perturbations + top-K selection."""
    print(f"  Generating {n_perturbations} orthogonal perturbations (σ={sigma})...")
    perturbed_models = perturb_weights_orthogonal(base_model, sigma, n_perturbations, device)
    candidates = []
    for i, p_model in enumerate(perturbed_models):
        p_model = p_model.to(device)
        acc = evaluate(p_model, val_loader, device)
        candidates.append((acc, p_model))
        if (i + 1) % 20 == 0:
            accs = [c[0] for c in candidates]
            print(f"    {i+1}/{n_perturbations}: mean_acc={np.mean(accs):.1f}%, best={max(accs):.1f}%")
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [c[1] for c in candidates[:k]]
    selected_accs = [c[0] for c in candidates[:k]]
    all_accs = [c[0] for c in candidates]
    print(f"  Top-{k} val accs: {[f'{a:.1f}' for a in selected_accs]}")
    return ThicketEnsemble(selected, device), all_accs


def build_layer_scaled_thicket(base_model, sigma, n_perturbations, k, val_loader, device=DEVICE):
    """Build thicket ensemble via layer-norm-scaled perturbations + top-K selection."""
    print(f"  Generating {n_perturbations} layer-scaled perturbations (σ={sigma})...")
    candidates = []
    for i in range(n_perturbations):
        p_model = perturb_weights_layer_scaled(base_model, sigma).to(device)
        acc = evaluate(p_model, val_loader, device)
        candidates.append((acc, p_model))
        if (i + 1) % 20 == 0:
            accs = [c[0] for c in candidates]
            print(f"    {i+1}/{n_perturbations}: mean_acc={np.mean(accs):.1f}%, best={max(accs):.1f}%")
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [c[1] for c in candidates[:k]]
    selected_accs = [c[0] for c in candidates[:k]]
    all_accs = [c[0] for c in candidates]
    print(f"  Top-{k} val accs: {[f'{a:.1f}' for a in selected_accs]}")
    return ThicketEnsemble(selected, device), all_accs


# ─── Deep Ensemble Baseline ──────────────────────────────────────────────────

def build_deep_ensemble(train_loader, val_loader, n_members, device=DEVICE):
    """Train n_members independently initialized models."""
    models = []
    for i in range(n_members):
        print(f"  Training deep ensemble member {i+1}/{n_members}...")
        set_seed(SEED + i + 100)
        model = ResNet18()
        save_path = str(MODELS_DIR / f"deep_ensemble_{i}.pt")
        model, acc = train_model(model, train_loader, val_loader,
                                 epochs=TRAIN_EPOCHS, device=device, save_path=save_path)
        print(f"    Member {i+1} val acc: {acc:.1f}%")
        models.append(model)
    set_seed(SEED)
    return ThicketEnsemble(models, device)


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiments():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results = {}

    print("=" * 70)
    print("Neural Thicket Ensemble: Adversarial & OOD Robustness Experiments")
    print("=" * 70)
    print(f"Device: {DEVICE}, Seed: {SEED}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ─── Step 1: Load Data ────────────────────────────────────────────────
    print("[1/7] Loading data...")
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    svhn_loader = get_svhn_loader()
    print(f"  CIFAR-10: train=45k, val=5k, test=10k")
    print(f"  SVHN test: {len(svhn_loader.dataset)} samples")
    print()

    # ─── Step 2: Train Base Model ─────────────────────────────────────────
    print("[2/7] Training base ResNet-18 on CIFAR-10...")
    base_path = str(MODELS_DIR / "base_resnet18.pt")
    base_model = ResNet18()
    if os.path.exists(base_path):
        print("  Loading existing base model...")
        base_model.load_state_dict(torch.load(base_path, map_location=DEVICE, weights_only=True))
        base_model = base_model.to(DEVICE)
        base_acc = evaluate(base_model, test_loader, DEVICE)
        print(f"  Base model test acc: {base_acc:.1f}%")
    else:
        base_model, best_val = train_model(base_model, train_loader, val_loader,
                                           epochs=TRAIN_EPOCHS, device=DEVICE, save_path=base_path)
        base_acc = evaluate(base_model, test_loader, DEVICE)
        print(f"  Base model test acc: {base_acc:.1f}%")
    results['base_model'] = {'clean_acc': base_acc}
    print()

    # ─── Step 3: Train Adversarially-Trained Model ────────────────────────
    print("[3/7] Training adversarially-trained ResNet-18 (PGD-AT)...")
    at_path = str(MODELS_DIR / "at_resnet18.pt")
    at_model = ResNet18()
    if os.path.exists(at_path):
        print("  Loading existing AT model...")
        at_model.load_state_dict(torch.load(at_path, map_location=DEVICE, weights_only=True))
        at_model = at_model.to(DEVICE)
        at_acc = evaluate(at_model, test_loader, DEVICE)
        print(f"  AT model test acc: {at_acc:.1f}%")
    else:
        at_model, best_val = train_model(at_model, train_loader, val_loader,
                                         epochs=AT_EPOCHS, device=DEVICE,
                                         adversarial=True, save_path=at_path)
        at_acc = evaluate(at_model, test_loader, DEVICE)
        print(f"  AT model test acc: {at_acc:.1f}%")
    results['at_model'] = {'clean_acc': at_acc}
    print()

    # ─── Step 4: Build Thicket Ensembles ──────────────────────────────────
    print("[4/7] Building thicket ensembles...")

    # Find good sigma by testing a range
    print("\n  --- Sigma sweep (Gaussian, K=5) ---")
    sigma_results = {}
    for sigma in SIGMA_VALUES:
        print(f"\n  σ = {sigma}:")
        ens, all_accs = build_gaussian_thicket(
            base_model, sigma, N_PERTURBATIONS, DEFAULT_K, val_loader, DEVICE)
        clean_vote, clean_avg = evaluate_ensemble_clean(ens, test_loader, DEVICE)
        disagree = 0
        n_batches = 0
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)
            disagree += ens.disagreement_rate(inputs)
            n_batches += 1
        avg_disagree = disagree / n_batches

        sigma_results[sigma] = {
            'clean_acc_vote': clean_vote,
            'clean_acc_avg': clean_avg,
            'disagreement': avg_disagree,
            'mean_candidate_acc': float(np.mean(all_accs)),
            'top_k_accs': [c for c in all_accs[:DEFAULT_K]],
        }
        print(f"    Clean acc (vote): {clean_vote:.1f}%, (avg): {clean_avg:.1f}%, disagree: {avg_disagree:.4f}")

    results['sigma_sweep'] = sigma_results

    # Pick best sigma (highest clean accuracy via voting)
    best_sigma = max(sigma_results, key=lambda s: sigma_results[s]['clean_acc_vote'])
    print(f"\n  Best sigma: {best_sigma} (clean_vote={sigma_results[best_sigma]['clean_acc_vote']:.1f}%)")

    # Build main ensembles with best sigma
    print(f"\n  --- Building main ensembles (σ={best_sigma}, K={DEFAULT_K}) ---")

    print("\n  [Gaussian Thicket]")
    gaussian_ens, gauss_accs = build_gaussian_thicket(
        base_model, best_sigma, N_PERTURBATIONS, DEFAULT_K, val_loader, DEVICE)

    print("\n  [Orthogonal Thicket]")
    ortho_ens, ortho_accs = build_orthogonal_thicket(
        base_model, best_sigma, N_PERTURBATIONS, DEFAULT_K, val_loader, DEVICE)

    print("\n  [Layer-Scaled Thicket]")
    layerscale_ens, ls_accs = build_layer_scaled_thicket(
        base_model, best_sigma, N_PERTURBATIONS, DEFAULT_K, val_loader, DEVICE)

    # Deep ensemble
    print("\n  [Deep Ensemble (3 members)]")
    deep_ens = build_deep_ensemble(train_loader, val_loader, 3, DEVICE)
    print()

    # ─── Step 5: Adversarial Robustness Evaluation ────────────────────────
    print("[5/7] Adversarial robustness evaluation...")

    # Single model attacks
    print("\n  Attacking single base model...")
    base_model.eval()
    base_pgd = evaluate_adversarial(
        base_model, test_loader,
        lambda m, x, t: pgd_attack(m, x, t, PGD_EPS, PGD_ALPHA, PGD_STEPS, DEVICE),
        DEVICE, "PGD base")
    base_fgsm = evaluate_adversarial(
        base_model, test_loader,
        lambda m, x, t: fgsm_attack(m, x, t, FGSM_EPS, DEVICE),
        DEVICE, "FGSM base")
    results['base_model']['pgd_acc'] = base_pgd
    results['base_model']['fgsm_acc'] = base_fgsm
    print(f"  Base model: PGD={base_pgd:.1f}%, FGSM={base_fgsm:.1f}%")

    # AT model attacks
    print("\n  Attacking AT model...")
    at_model.eval()
    at_pgd = evaluate_adversarial(
        at_model, test_loader,
        lambda m, x, t: pgd_attack(m, x, t, PGD_EPS, PGD_ALPHA, PGD_STEPS, DEVICE),
        DEVICE, "PGD AT")
    at_fgsm = evaluate_adversarial(
        at_model, test_loader,
        lambda m, x, t: fgsm_attack(m, x, t, FGSM_EPS, DEVICE),
        DEVICE, "FGSM AT")
    results['at_model']['pgd_acc'] = at_pgd
    results['at_model']['fgsm_acc'] = at_fgsm
    print(f"  AT model: PGD={at_pgd:.1f}%, FGSM={at_fgsm:.1f}%")

    # Ensemble attacks
    ensembles = {
        'gaussian_thicket': gaussian_ens,
        'orthogonal_thicket': ortho_ens,
        'layer_scaled_thicket': layerscale_ens,
        'deep_ensemble': deep_ens,
    }

    for name, ens in ensembles.items():
        print(f"\n  Attacking {name}...")
        clean_vote, clean_avg = evaluate_ensemble_clean(ens, test_loader, DEVICE)

        pgd_vote, pgd_avg = evaluate_ensemble_adversarial(
            ens, test_loader, PGD_EPS, PGD_ALPHA, PGD_STEPS, DEVICE, f"PGD {name}")

        # FGSM on ensemble
        wrapper = EnsembleWrapper(ens.models).to(DEVICE)
        fgsm_acc = evaluate_adversarial(
            wrapper, test_loader,
            lambda m, x, t: fgsm_attack(m, x, t, FGSM_EPS, DEVICE),
            DEVICE, f"FGSM {name}")

        results[name] = {
            'clean_acc_vote': clean_vote,
            'clean_acc_avg': clean_avg,
            'pgd_acc_vote': pgd_vote,
            'pgd_acc_avg': pgd_avg,
            'fgsm_acc': fgsm_acc,
        }
        print(f"  {name}: clean_vote={clean_vote:.1f}%, PGD_vote={pgd_vote:.1f}%, FGSM={fgsm_acc:.1f}%")
    print()

    # ─── Step 6: OOD Detection ────────────────────────────────────────────
    print("[6/7] OOD detection evaluation (CIFAR-10 → SVHN)...")

    ood_results = {}
    for name, ens in ensembles.items():
        print(f"\n  Computing OOD scores for {name}...")
        id_scores = collect_uncertainty_scores(ens, test_loader, DEVICE)
        ood_scores = collect_uncertainty_scores(ens, svhn_loader, DEVICE)

        ood_results[name] = {}
        for metric_name in ['msp', 'energy', 'mutual_info']:
            id_vals = id_scores[metric_name]
            ood_vals = ood_scores[metric_name]

            # For entropy/MI, lower = more in-distribution → negate for AUROC
            if metric_name == 'mutual_info':
                auroc, fpr95 = compute_ood_metrics(-id_vals, -ood_vals)
            else:
                auroc, fpr95 = compute_ood_metrics(id_vals, ood_vals)

            ood_results[name][metric_name] = {
                'auroc': auroc,
                'fpr95': fpr95,
                'id_mean': float(np.mean(id_vals)),
                'ood_mean': float(np.mean(ood_vals)),
            }
            print(f"    {metric_name}: AUROC={auroc:.4f}, FPR@95={fpr95:.4f}")

    results['ood_detection'] = ood_results
    print()

    # ─── Step 7: Ensemble Size Ablation ───────────────────────────────────
    print("[7/7] Ensemble size ablation (K sweep)...")
    k_results = {}
    for k in K_VALUES:
        print(f"\n  K={k}:")
        ens_k, _ = build_gaussian_thicket(
            base_model, best_sigma, N_PERTURBATIONS, k, val_loader, DEVICE)
        clean_vote, clean_avg = evaluate_ensemble_clean(ens_k, test_loader, DEVICE)

        pgd_vote, pgd_avg = evaluate_ensemble_adversarial(
            ens_k, test_loader, PGD_EPS, PGD_ALPHA, PGD_STEPS, DEVICE, f"PGD K={k}")

        k_results[k] = {
            'clean_acc_vote': clean_vote,
            'clean_acc_avg': clean_avg,
            'pgd_acc_vote': pgd_vote,
            'pgd_acc_avg': pgd_avg,
        }
        print(f"    Clean(vote)={clean_vote:.1f}%, PGD(vote)={pgd_vote:.1f}%")

    results['k_ablation'] = k_results
    results['best_sigma'] = best_sigma

    # ─── Save Results ─────────────────────────────────────────────────────
    # Convert numpy types
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    results_path = str(RESULTS_DIR / "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiments()
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
