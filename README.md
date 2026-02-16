**Aim**: To develop a robust Bayesian Vector Autoregression (BVAR) framework for multivariate time-series forecasting that explicitly accounts for distributional uncertainty in innovation processes. The proposed framework integrates Wasserstein distributionally robust optimization (DRO) with diffusion-based generative modeling of VAR residuals, enabling stable forecasting, calibrated uncertainty quantification, and systematic stress testing under distributional shifts and rare events. Rather than assuming Gaussian or parametric innovation distributions, the framework learns a flexible, nonparametric residual law via diffusion processes and constructs ambiguity sets around this learned distribution to evaluate worst-case forecasts and impulse responses.

**Keywords**: _Bayesian Vector Autoregression, Distributionally Robust Optimization, Wasserstein Distance, Diffusion Models, Robust Innovations, Non-Gaussian Shocks, Multivariate Time Series, Uncertainty Quantification, Stress Testing, Bayesian Priors_

**Research Questions**
  - How can Bayesian VARs be reformulated to remain stable and informative under uncertainty about the innovation distribution, rather than relying on Gaussian or parametric assumptions?
  - How does Wasserstein-based distributional robustness alter predictive uncertainty, forecast calibration, and tail behavior in multivariate time-series models?
  - Can diffusion-based generative models trained on VAR residuals provide a flexible and data-driven alternative to Gaussian or bootstrap innovation assumptions?
  - To what extent do diffusion-generated innovations capture tail risk, rare events, and stress scenarios more effectively than classical residual models?
  - How do shrinkage priors for VAR coefficients interact with distributionally robust innovation modeling in determining forecast accuracy and robustness?
  - How can a robust BVAR framework be used to construct systematic stress tests and worst-case scenario analyses under controlled distributional perturbations?

**Objectives**
  - Construct a general-purpose multivariate BVAR framework in which robustness to innovation misspecification is incorporated via Wasserstein distributionally robust optimization.
  - Formulate ambiguity sets centered on learned innovation distributions and apply duality results from DRO to derive worst-case predictive functionals for BVAR forecasts.
  - Develop diffusion-based generative models for VAR residuals, treating innovations as samples from a learned nonparametric distribution rather than a fixed parametric family.
  - Integrate diffusion-based innovation modeling into the BVAR–DRO pipeline while maintaining linear VAR dynamics and interpretability of impulse responses.
  - Systematically compare classical BVARs, DRO-regularized BVARs with parametric innovations, and DRO-BVARs augmented with diffusion-based innovations using forecast accuracy, predictive coverage, tail-risk metrics, and stress-test outcomes.
  - Design a domain-agnostic stress-testing framework that evaluates forecast sensitivity to distributional shifts and rare-event shocks without reliance on application-specific structural assumptions.
  - Deliver an open-source software package implementing robust BVAR estimation, diffusion-based innovation modeling, and DRO-based forecasting and stress testing.
