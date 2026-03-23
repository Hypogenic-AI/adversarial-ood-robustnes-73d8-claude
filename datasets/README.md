# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: CIFAR-10

### Overview
- **Source**: torchvision.datasets.CIFAR10
- **Size**: 50,000 train + 10,000 test samples, ~170MB per split
- **Format**: PyTorch dataset (pickle files)
- **Task**: 10-class image classification (32x32 RGB)
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **License**: MIT

### Download Instructions

```python
import torchvision
torchvision.datasets.CIFAR10(root='datasets/cifar10', train=True, download=True)
torchvision.datasets.CIFAR10(root='datasets/cifar10', train=False, download=True)
```

### Loading
```python
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trainset = torchvision.datasets.CIFAR10(root='datasets/cifar10', train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='datasets/cifar10', train=False, transform=transform)
```

### Notes
- Standard benchmark for adversarial robustness evaluation
- ℓ∞ perturbation budget: ε=8/255 (standard)
- ℓ2 perturbation budget: ε=128/255 (standard)
- Used by: ADP (Pang et al. 2019), Model Soups (Croce et al. 2023), Split-Ensemble

---

## Dataset 2: CIFAR-100

### Overview
- **Source**: torchvision.datasets.CIFAR100
- **Size**: 50,000 train + 10,000 test samples, ~170MB per split
- **Format**: PyTorch dataset (pickle files)
- **Task**: 100-class image classification (32x32 RGB)
- **Classes**: 100 fine-grained classes in 20 superclasses
- **License**: MIT

### Download Instructions

```python
import torchvision
torchvision.datasets.CIFAR100(root='datasets/cifar100', train=True, download=True)
torchvision.datasets.CIFAR100(root='datasets/cifar100', train=False, download=True)
```

### Loading
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
trainset = torchvision.datasets.CIFAR100(root='datasets/cifar100', train=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='datasets/cifar100', train=False, transform=transform)
```

### Notes
- More challenging than CIFAR-10; richer diversity structure
- Used by: ADP, Split-Ensemble, diversity regularization papers
- Useful for evaluating how class count affects thicket diversity

---

## Dataset 3: SVHN (Street View House Numbers)

### Overview
- **Source**: torchvision.datasets.SVHN
- **Size**: 26,032 test samples, ~62MB
- **Format**: PyTorch dataset (MATLAB .mat files)
- **Task**: 10-class digit recognition (32x32 RGB)
- **License**: Non-commercial research

### Download Instructions

```python
import torchvision
torchvision.datasets.SVHN(root='datasets/svhn', split='test', download=True)
```

### Loading
```python
transform = transforms.Compose([transforms.ToTensor()])
svhn_test = torchvision.datasets.SVHN(root='datasets/svhn', split='test', transform=transform)
```

### Notes
- Standard OOD dataset when CIFAR-10 is the in-distribution dataset
- Same resolution (32x32) as CIFAR → fair comparison
- Well-separated from CIFAR distribution (digits vs natural images)
- Used by Split-Ensemble and many OOD detection papers

---

## Usage in Experiments

### Adversarial Robustness Evaluation
- Train models on CIFAR-10/100 train split
- Evaluate clean accuracy on test split
- Generate adversarial examples using PGD (ℓ∞, ε=8/255, 20-50 steps)
- Evaluate robust accuracy on adversarial test set

### OOD Detection Evaluation
- Train on CIFAR-10 (in-distribution)
- Evaluate OOD detection using SVHN test set (out-of-distribution)
- Metrics: AUROC, FPR@95
- Compare uncertainty scores: MSP, Energy, Entropy, MI
