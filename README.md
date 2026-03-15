# droBVAR

This repository hosts the  project for `droBVAR` development, A robust Bayesian VAR framework with diffusion-based innovation modeling

**Author:** Jonathan Ma

**Advisor:** Dr. Luhao Zhang

---

**Aim**: To develop a robust Bayesian Vector Autoregression (BVAR) framework for multivariate time-series forecasting that explicitly accounts for distributional uncertainty in innovation processes. The proposed framework integrates Wasserstein distributionally robust optimization (DRO) with diffusion-based generative modeling of VAR residuals, enabling stable forecasting, calibrated uncertainty quantification, and systematic stress testing under distributional shifts and rare events. Rather than assuming Gaussian or parametric innovation distributions, the framework learns a flexible, nonparametric residual law via diffusion processes and constructs ambiguity sets around this learned distribution to evaluate worst-case forecasts and impulse responses.

**Keywords**: _Bayesian Vector Autoregression, Distributionally Robust Optimization, Wasserstein Distance, Diffusion Models, Robust Innovations, Non-Gaussian Shocks, Multivariate Time Series, Uncertainty Quantification, Stress Testing, Bayesian Priors_

---

## Data Source

**Dataset:** TBD

---

## Methodology

TBD [notes here](https://www.overleaf.com/read/ncfnzvcchcbv#a2fb3c)

---


## Structure

```text
innovative-droBVAR/
├── code/                        
│   └── requirements.txt          # dependencies
├── data/
│   └── raw/                      # Immutable input datasets
│   └── cleaned/                  # Versions of cleaned data
├── docs/                         
│   ├── CHANGELOG.md              # Project updates and version history
│   └── GithubRef.md              # Quick reference for GitHub Usage
├── drobvar/                      # Package Development
│   └── main.py                   # Notes from readings
├── refs/                         # My learning journey         
│   ├── 01_var_mechanics.ipynb    
│   ├── 02_bvar_mechanics.ipynb    
│   ├── 03_innovation_modeling.ipynb   
│   ├── 04_diffusion_residual_generator.ipynb    
│   └── 05_dro_forecasting.ipynb    
├── .gitignore                    # Files and folders excluded from Git tracking
├── LICENSE                       # Usage license
└── README.md                     # Project overview 
```
