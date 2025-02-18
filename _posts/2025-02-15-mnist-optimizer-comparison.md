---
layout: post
title: "Kiefer-Wolfowitz Algorithm for Gradient Approximation"
date: 2025-02-15 16:15:49
author: rg625
---

## Introduction

This blog post examines the application of the Kiefer-Wolfowitz (KW) algorithm for gradient approximation in both optimization and sampling contexts. The first application involves utilizing the KW method to optimize a shallow neural network for linear regression and a convolutional neural network (CNN) for classification. The second application leverages the KW algorithm to approximate the score of a probability distribution.

## Optimization

The performance of the KW-based optimizer is compared against several state-of-the-art optimization algorithms commonly employed in the literature, including Adagrad, Adam, and Stochastic Gradient Descent (SGD). Additionally, it is contrasted with a similar approach that incorporates random perturbation, known as Simultaneous Perturbation Stochastic Approximation (SPSA). Two variants of the KW optimizer were implemented: a basic version adhering strictly to the theoretical framework and an adaptive variant that incorporates the adaptive step-size mechanism from Adam.

### Linear Regression

The shallow neural network utilized for linear regression consists of a single layer with 20 trainable weights. The dataset was synthetically generated using a randomly sampled set of weights. The performance of each optimization algorithm was evaluated based on training loss, test loss, training accuracy, test accuracy, computational cost, complexity, and training time. The results are presented in the figures below.

<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">
  <iframe src="https://rg625.github.io/assets/images/kw/linear_regression/metrics_evolution.html" width="100%" height="600px" frameborder="0"></iframe>
  <iframe src="https://rg625.github.io/assets/images/kw/linear_regression/parameters_evolution.html" width="100%" height="600px" frameborder="0"></iframe>
</div>

### Classification

A CNN model was trained using various optimizers on the MNIST dataset. The model architecture comprises two convolutional layers with Rectified Linear Unit (ReLU) activation and max pooling, followed by two fully connected layers with ReLU activation, amounting to a total of 55,338 trainable parameters. The model was trained for 10 epochs with a batch size of 1024 and a learning rate of 0.01. Performance was assessed in terms of training loss, test loss, training accuracy, test accuracy, computational cost, complexity, and training duration.

## Results

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; width: 100%;">
  <div>
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/comparative_train_loss.png" alt="Train Losses" width="100%">
  </div>
  <div>
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/comparative_test_accuracy.png" alt="Test Accuracies" width="100%">
  </div>
  <div>
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/comparative_compute_cost.png" alt="Computational Costs" width="100%">
  </div>
  <div>
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/train_times.png" alt="Training Times" width="100%">
  </div>
</div>

### Predictions and Confusion Matrices

The figures below illustrate the predictions and confusion matrices for each optimizer. Each figure presents a 4x4 grid of randomly selected test samples, along with their true and predicted labels.

<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">
  <iframe src="https://rg625.github.io/assets/images/kw/mnist_classifier/predictions.html" width="100%" height="600px" frameborder="0"></iframe>
  <iframe src="https://rg625.github.io/assets/images/kw/mnist_classifier/confusion_matrix.html" width="100%" height="600px" frameborder="0"></iframe>
</div>

## Sampling

The second application of the KW method involves estimating the score function of a probability distribution for use in two Markov Chain Monte Carlo (MCMC) sampling algorithms: Langevin Dynamics (LD) and Hamiltonian/Hybrid Monte Carlo (HMC). The performance of KW in this context was compared against PyTorch’s autodifferentiation-based approach and Spall’s SPSA method. For a comprehensive analysis, all these models were tested for different numbers of steps in the MCMC chains and samples. The main results displaying the evolution of a particular sample through the MCMC chain and the empirically found distribution are below, along with a comparison of memory usages and runtimes. More results are displayed on the project's page.

<div style="display: flex; flex-direction: column; gap: 20px; width: 100%;">
  <iframe src="https://rg625.github.io/assets/images/kw/kw_sampling/convergence_plot_1_1000.html" width="100%" height="600px" frameborder="0"></iframe>
  <iframe src="https://rg625.github.io/assets/images/kw/kw_sampling/histogram_comparison_10000_1000.html" width="100%" height="600px" frameborder="0"></iframe>
</div>

## Conclusion

The results indicate that different optimization algorithms exhibit varying levels of performance in terms of training loss, test loss, accuracy, computational cost, complexity, and training duration. The Adam optimizer consistently demonstrates strong performance, while other optimizers, such as SPSA and Kiefer-Wolfowitz, yield competitive results depending on the metric under consideration. The KW algorithm, however, is significantly slower than autodifferentiation-based optimizers, as it requires perturbing each parameter individually to estimate the gradient direction. This leads to a linear scaling in computational cost with respect to the number of parameters, making KW impractical for large-scale optimization tasks. While KW exhibits superior convergence properties compared to Adam on average, its computational inefficiency renders it infeasible for practical use. In contrast, SPSA provides a faster alternative at the cost of reduced accuracy.

Regarding the second application, the KW method demonstrates comparable performance to autodifferentiation in score estimation; however, it requires additional hyperparameter tuning, particularly in selecting the appropriate perturbation parameter.

The source code for these experiments is available on my [GitHub repository](https://github.com/rg625/KW_grad_estimation). Readers are encouraged to explore the implementation and reproduce the results.

