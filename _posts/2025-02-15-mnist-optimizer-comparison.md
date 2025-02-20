---
layout: post
title: "Kiefer-Wolfowitz Algorithm for Gradient Approximation: Theoretical and Empirical Analysis"
date: 2025-02-15 16:15:49
author: rg625
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Introduction

This article provides a rigorous analysis of the Kiefer-Wolfowitz (KW) algorithm as a stochastic gradient approximation method and its application in optimization and score function estimation. Specifically, we explore the KW method’s convergence properties, computational complexity, and practical utility in optimizing neural networks and estimating score functions for sampling-based inference. We provide a formal derivation of the KW estimator, compare its empirical performance to state-of-the-art optimization algorithms such as Adagrad, Adam, and Stochastic Gradient Descent (SGD), and analyze its efficacy relative to Simultaneous Perturbation Stochastic Approximation (SPSA). Additionally, we investigate its role in Markov Chain Monte Carlo (MCMC) sampling algorithms such as Langevin Dynamics (LD) and Hamiltonian Monte Carlo (HMC). 

## The Kiefer-Wolfowitz Gradient Approximation

The KW algorithm is a finite-difference method for gradient estimation, expressed as follows for a function $$f: \mathbb{R}^d \to \mathbb{R}$$:

$$
\nabla f(\theta) \approx \frac{f(\theta + c e_i) - f(\theta - c e_i)}{2c} e_i,
$$

where $$e_i$$ is the unit vector along the $$i$$-th coordinate, and $$c > 0$$ is a perturbation parameter. Given $$d$$ parameters, the full gradient estimate requires $$2d$$ function evaluations, rendering it computationally expensive for high-dimensional problems. The update rule for parameter estimation follows:

$$
\theta_{t+1} = \theta_t - \alpha_t \hat{\nabla} f(\theta_t),
$$

where $$\alpha_t$$ is a step-size sequence satisfying the Robbins-Monro conditions:

$$
\sum_{t=1}^{\infty} \alpha_t = \infty, \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty.
$$

### Convergence Analysis

Under standard smoothness assumptions on $$f(\theta)$$, the KW method exhibits asymptotic unbiasedness in gradient estimation. Specifically, if $$f$$ is differentiable with Lipschitz gradient, the KW estimate satisfies:

$$
\mathbb{E}[\hat{\nabla} f(\theta)] = \nabla f(\theta) + \mathcal{O}(c^2).
$$

This error bound suggests that smaller perturbations yield more accurate gradients at the cost of increased variance. Convergence in expectation to a stationary point follows from classical stochastic approximation theory. 

## Application to Optimization

### Linear Regression with a Shallow Neural Network

A single-layer neural network with $$d = 20$$ trainable parameters was optimized using KW, Adam, Adagrad, SGD, and SPSA. The synthetic dataset was generated from a true weight vector $$w^* \in \mathbb{R}^{d}$$, and training was performed using mean squared error (MSE) loss:

$$
\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} (y_i - x_i^T w)^2.
$$

Performance metrics included training loss, test loss, computational complexity, and convergence speed. Results are presented in the following figures:

<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">
  <iframe src="https://rg625.github.io/assets/images/kw/linear_regression/metrics_evolution.html" width="100%" height="600px" frameborder="0"></iframe>
  <iframe src="https://rg625.github.io/assets/images/kw/linear_regression/parameters_evolution.html" width="100%" height="600px" frameborder="0"></iframe>
</div>

### Classification Using a Convolutional Neural Network

A CNN with 55,338 trainable parameters was trained on the MNIST dataset using various optimizers. The architecture consisted of two convolutional layers (ReLU activation, max pooling) followed by two fully connected layers:

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{n} y_i \log \hat{y}_i,
$$

where $$y_i$$ is the true label and $$\hat{y}_i$$ is the predicted probability. The model was trained for 10 epochs with a batch size of 1024 and a learning rate of 0.01.

## Results

<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">
  <iframe src="https://rg625.github.io/assets/images/kw/mnist_classifier/optimizer_results.html" width="100%" height="600px" frameborder="0"></iframe>
</div>

### Predictions and Confusion Matrices

<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">
  <iframe src="https://rg625.github.io/assets/images/kw/mnist_classifier/predictions.html" width="100%" height="600px" frameborder="0"></iframe>
  <iframe src="https://rg625.github.io/assets/images/kw/mnist_classifier/confusion_matrix.html" width="100%" height="600px" frameborder="0"></iframe>
</div>

## Application to Sampling: Score Function Estimation

The KW method was used to approximate the score function $$\nabla \log p(x)$$ for use in Langevin Dynamics and HMC sampling. The score function estimation follows:

$$
\hat{\nabla \log p(x)} = \frac{p(x + c e_i) - p(x - c e_i)}{2c} e_i.
$$

Empirical comparisons with PyTorch’s autodifferentiation and SPSA showed that KW performs competitively but requires hyperparameter tuning for optimal perturbation selection.

<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">
  <iframe src="https://rg625.github.io/assets/images/kw/kw_sampling/convergence_plot_1_1000.html" width="100%" height="600px" frameborder="0"></iframe>
  <iframe src="https://rg625.github.io/assets/images/kw/kw_sampling/histogram_comparison_10000_1000.html" width="100%" height="600px" frameborder="0"></iframe>
</div>

## Conclusion

While KW demonstrates strong theoretical convergence guarantees, its computational inefficiency due to the linear scaling of function evaluations with dimensionality makes it impractical for large-scale optimization. KW’s gradient estimates are asymptotically unbiased but suffer from high variance, requiring careful selection of the perturbation parameter. Comparisons with Adam, Adagrad, and SPSA indicate that while KW achieves competitive loss minimization, its computational burden outweighs its benefits. For score function estimation, KW provides a viable alternative to autodifferentiation but necessitates extensive hyperparameter tuning.

The source code is available on my [GitHub repository](https://github.com/rg625/KW_grad_estimation), allowing for reproducibility and further exploration of these findings.

