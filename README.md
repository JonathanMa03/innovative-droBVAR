# Calibration-Aware Diffusion Residual Learning for Uncertainty Quantification Under Residual Misspecification

> Can Calibration feedback be incorporated directly into diffusion-based residual learning to produce more reliable and robust uncertainty estimates than standard deep residual models like ResNets?

Modern diffusion models provide flexible, nonparametric representations of uncertainty and have shown promise for modeling complex residual distributions. However, diffusion residual models are typically trained solely through denoising or score-matching objectives, with calibration assessed only after training. A diffusion model may accurately learn the residual distribution while still generating predictive uncertainty that is poorly calibrated. This thesis proposes Calibration-Aware Diffusion Residual Learning (CADRL), a framework that incorporates calibration directly into diffusion training through a calibration-aware learning objective.

Rather than treating calibration as a post-hoc evaluation metric, calibration becomes a training signal that guides the residual learning process itself.

## Research Questions 

- Can calibration feedback improve diffusion-based residual learning?
- Which calibration objectives (ECE, PIT, CRPS, Energy, Coverage) are most effective as learning signals?
- Does calibration-aware training improve uncertainty quantification under innovation misspecification?
- How does calibration-aware diffusion compare with standard diffusion residual learning?
- How robust are calibration-aware diffusion residual models under distributional perturbations and structural shifts?

## Methodological Contribution

The primary contribution shifts from employing diffusion models to a new learning procedure:

$$\mathcal{L}=\mathcal{L}_{\text{diffusion}}\quad \rightarrow \quad \mathcal{L}=\mathcal{L}_{\text{diffusion}}+\lambda \mathcal{L}_{\text{calibration}}$$

The first iteration will rely on Vector Autoregressions and Recurrent Neural Networks to investigate how CARL affects linear and nonlinear baselines. The innovation models used for testing are a Gaussian baseline, heavy-tail behavior through Student-T, mixture innovaotions, and time-varying innovations. We will investigate variance inflation, tail amplification, mixture contamination, and regime shifts using degradation metrics, and then the framework will be applied to financial return data

## Long Term Research Vision

The first iteration develops a calibration-aware diffusion implementation for the broader CARL (Calibration-Aware Residual Learning) Framework where the learner can be replaced with mflow models or Bayesian nonparametric models. Also will consider extensions to Images and computer vision, as the ultimate goal is to become model-agnostic.