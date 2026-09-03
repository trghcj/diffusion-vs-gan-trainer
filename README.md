# Diffusion-vs-GAN Trainer with CI

A rigorous, production-grade generative modeling laboratory comparing a **Time-Conditioned Residual Diffusion Model (DDPM/DDIM)** against a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** baseline.

Designed with continuous integration (CI) evidence at its core: mathematical unit tests, invariant assertions, capacity-matched architectures, automated training smoke runs, and quantitative evaluation reports (FID/KID) with publication-standard qualitative grids.

---

## 1. System Overview & Architectural Design

Both models are trained on identical preprocessed data with exact parameter balancing, unified $[-1, 1]$ tensor normalization invariants, and evaluated via an impartial feature extraction pipeline.

```
                     [ Preprocessed Dataset (Fashion-MNIST / CIFAR-10 / Synthetic) ]
                                                    │
                      ┌─────────────────────────────┴─────────────────────────────┐
                      ▼                                                           ▼
         [ Diffusion Model (U-Net) ]                                 [ WGAN-GP Baseline ]
         ├── Forward Diffusion q(x_t|x_0)                            ├── Generator G(z): z ~ N(0, I) -> x_fake
         ├── Epsilon Prediction Network                              └── Critic D(x): 1-Lipschitz (GroupNorm)
         └── DDIM Fast Sampler (20-50 steps)                         └── Gradient Penalty ||grad(D)||_2 ~ 1
                      │                                                           │
                      ▼                                                           ▼
         [ Checkpoint: diffusion.pt ]                                [ Checkpoint: wgan.pt ]
                      │                                                           │
                      └─────────────────────────────┬─────────────────────────────┘
                                                    ▼
                                      [ Evaluation Engine ]
                                      ├── Feature Extractor (Efficient CNN / InceptionV3)
                                      ├── Quantitative Metrics (FID, KID)
                                      ├── Qualitative Visual Grid (Real vs Diff vs GAN)
                                      └── Export: metrics_summary.json & comparison_report.md
```

---

## 2. Mathematical Foundations

### 2.1 Diffusion Formulation (DDPM & DDIM)

1. **Forward Process (Markov chain with Gaussian transitions)**:
   $$q(x_t | x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
   where $\beta_t \in (0, 1)$, $\alpha_t = 1 - \beta_t$, and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

2. **Training Objective (Simplified Score Matching)**:
   The network $\epsilon_\theta(x_t, t)$ is optimized using mean squared error against ground-truth Gaussian noise:
   $$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

3. **Inference / Sampling**:
   - **DDPM (Ancestral Markov Sampling)**: Reverses the diffusion chain over $T$ discrete steps ($T=1000$).
   - **DDIM (Deterministic Accelerated Sampling)**: Non-Markovian forward process enabling high-fidelity sampling in $S \ll T$ steps (typically $S \in [20, 50]$):
     $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\epsilon_\theta(x_t, t) + \sigma_t \epsilon$$
     Setting $\eta = 0 \implies \sigma_t = 0$ yields a deterministic trajectory.

### 2.2 WGAN-GP Formulation

1. **Wasserstein-1 Objective (Kantorovich-Rubinstein Duality)**:
   $$\min_G \max_{D \in \mathcal{D}_L} \mathbb{E}_{x \sim \mathbb{P}_r}[D(x)] - \mathbb{E}_{\tilde{x} \sim \mathbb{P}_g}[D(\tilde{x})]$$

2. **Gradient Penalty (1-Lipschitz Regularization)**:
   To enforce the 1-Lipschitz condition without weight clipping artifacts, a gradient penalty is computed on interpolated samples $\hat{x} = \epsilon x + (1 - \epsilon)\tilde{x}$ for $\epsilon \sim U(0, 1)$:
   $$\mathcal{L}_D = \mathbb{E}_{\tilde{x}}[D(\tilde{x})] - \mathbb{E}_x[D(x)] + \lambda_{\text{gp}} \mathbb{E}_{\hat{x}}\left[ \left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2 \right]$$

---

## 3. Quantitative Evaluation Metrics

- **Fréchet Inception Distance (FID)**: Measures the 2-Wasserstein distance between multivariate Gaussians fitted to deep feature activations:
  $$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$
- **Kernel Inception Distance (KID)**: Computes the squared Maximum Mean Discrepancy (MMD) with a polynomial kernel $k(x, y) = \left(\frac{1}{d}x^T y + 1\right)^3$. KID is strictly unbiased and numerically stable on small evaluation subsets.

---

## 4. Repository Structure

```
.
├── .github/workflows/
│   └── ci.yml                 # Automated Ruff lint, Pytest coverage & End-to-End smoke pipeline
├── configs/
│   ├── diffusion_config.yaml  # Diffusion U-Net & schedule hyperparameters
│   ├── gan_config.yaml        # WGAN-GP generator/critic hyperparameters
│   └── eval_config.yaml       # Metric thresholds & sample sizes
├── src/genai_trainer/
│   ├── config.py              # Pydantic v2 configuration schemas
│   ├── cli.py                 # Unified CLI commands
│   ├── data/
│   │   ├── dataset.py         # Synthetic, Fashion-MNIST, and CIFAR-10 data loaders
│   │   └── transforms.py      # [-1, 1] range invariants & PIL utilities
│   ├── models/
│   │   ├── diffusion/         # Residual U-Net, Sinusoidal embeddings, Noise schedules
│   │   └── gan/               # WGAN Generator & Critic (GroupNorm)
│   ├── training/
│   │   ├── diffusion_trainer.py # DDPM/DDIM training loop
│   │   ├── gan_trainer.py       # WGAN-GP alternating training loop
│   │   └── checkpoint.py        # Safe atomic checkpoint manager
│   └── evaluation/
│       ├── metrics.py         # FID and KID computation engines
│       ├── visualizer.py      # Clean 3-row comparison grid renderer
│       └── reporter.py        # Markdown & JSON summary report writers
├── tests/
│   ├── test_data.py           # Invertible normalization & dataset checks
│   ├── test_diffusion_math.py # Noise schedule boundary & SNR monotonicity tests
│   ├── test_models.py         # Architecture shapes & gradient flow tests
│   ├── test_train_smoke.py    # 1-step CPU optimization smoke tests
│   └── test_eval_metrics.py   # Metric calculation & report generator tests
├── Makefile                   # Developer workflow commands
├── pyproject.toml             # Packaging, Ruff, and Pytest configuration
└── requirements.txt           # Pinned dependencies
```

---

## 5. Quickstart & Installation

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone https://github.com/your-org/diffusion-vs-gan-trainer.git
cd diffusion-vs-gan-trainer

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"
```

---

## 6. Usage & CLI Workflows

### 6.1 Run the Full Automated Pipeline
Train both models on the synthetic dataset, run evaluation, and output reports in seconds:

```bash
python -m genai_trainer.cli run-pipeline --dataset synthetic --epochs 2 --num-samples 32
```

### 6.2 Train the Diffusion Model Individually
```bash
python -m genai_trainer.cli train-diffusion --config configs/diffusion_config.yaml
```

### 6.3 Train the WGAN-GP Baseline Individually
```bash
python -m genai_trainer.cli train-gan --config configs/gan_config.yaml
```

### 6.4 Evaluate and Generate Comparison Report
```bash
python -m genai_trainer.cli evaluate --config configs/eval_config.yaml
```

Generated outputs will be saved to:
- `outputs/reports/metrics_summary.json`
- `outputs/reports/comparison_report.md`
- `outputs/reports/comparison_grid.png`

---

## 7. Testing & CI Pipeline

Run the comprehensive test suite locally:

```bash
# Run all unit and integration tests with coverage
pytest --cov=src/genai_trainer --cov-report=term-missing

# Run code hygiene checks
ruff check src tests
ruff format --check src tests
```

### GitHub Actions CI Job
The CI workflow defined in `.github/workflows/ci.yml` runs automatically on every push:
1. **Linter & Formatter**: Enforces PEP 8 compliance via Ruff.
2. **Unit Tests**: Runs 15+ automated tests verifying noise schedule math, parameter shapes, and metric calculations.
3. **End-to-End Smoke Test**: Executes 1 complete training epoch of Diffusion and WGAN-GP, computes FID/KID metrics, and archives the resulting report and visual grid as downloadable CI artifacts.
