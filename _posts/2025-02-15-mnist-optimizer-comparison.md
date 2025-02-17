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

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
  <div>
    <img src="https://rg625.github.io/assets/images/linear_optimizer_comparison/train_loss.png" alt="Train Loss" width="100%">
  </div>
  <div>
    <img src="https://rg625.github.io/assets/images/linear_optimizer_comparison/test_loss.png" alt="Test Loss" width="100%">
  </div>
  <div>
    <img src="https://rg625.github.io/assets/images/linear_optimizer_comparison/computational_cost.png" alt="Computational Cost" width="100%">
  </div>
  <div>
    <img src="https://rg625.github.io/assets/images/linear_optimizer_comparison/training_time.png" alt="Training Time" width="100%">
  </div>
</div>

### Classification

A CNN model was trained using various optimizers on the MNIST dataset. The model architecture comprises two convolutional layers with Rectified Linear Unit (ReLU) activation and max pooling, followed by two fully connected layers with ReLU activation, amounting to a total of 55,338 trainable parameters. The model was trained for 10 epochs with a batch size of 1024 and a learning rate of 0.01. Performance was assessed in terms of training loss, test loss, training accuracy, test accuracy, computational cost, complexity, and training duration.

## Results


<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
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

#### SPSA - Simple
<style>
  .grid-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    align-items: center; /* Ensures alignment */
  }
  .grid-item img {
    width: 100%;
    height: 100%; /* Forces images to take the full height */
    object-fit: contain; /* Ensures consistent scaling without distortion */
  }
  .grid-item {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 300px; /* Set a fixed height for equal sizing */
  }
</style>

<div class="grid-container">
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/predictions_SPSAOptimizerSimple.png" alt="Predictions - SPSAOptimizerSimple">
  </div>
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/confusion_matrix_SPSAOptimizerSimple.png" alt="Confusion Matrix - SPSAOptimizerSimple">
  </div>
</div>


#### SPSA - Adaptive

<div class="grid-container">
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/predictions_SPSAOptimizerAdaptive.png" alt="Predictions - SPSAOptimizerAdaptive">
  </div>
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/confusion_matrix_SPSAOptimizerAdaptive.png" alt="Confusion Matrix - SPSAOptimizerAdaptive">
  </div>
</div>

#### KW - Simple

<div class="grid-container">
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/predictions_KieferWolfowitzSimple.png" alt="Predictions - KieferWolfowitzSimple">
  </div>
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/confusion_matrix_KieferWolfowitzSimple.png" alt="Confusion Matrix - KieferWolfowitzSimple">
  </div>
</div>

#### KW - Adaptive

<div class="grid-container">
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/predictions_KieferWolfowitzAdaptive.png" alt="Predictions - KieferWolfowitzAdaptive">
  </div>
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/confusion_matrix_KieferWolfowitzAdaptive.png" alt="Confusion Matrix - KieferWolfowitzAdaptive">
  </div>
</div>

#### Adam

<div class="grid-container">
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/predictions_Adam.png" alt="Predictions - Adam">
  </div>
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/confusion_matrix_Adam.png" alt="Confusion Matrix - Adam">
  </div>
</div>

#### Adagrad

<div class="grid-container">
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/predictions_Adagrad.png" alt="Predictions - Adagrad">
  </div>
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/confusion_matrix_Adagrad.png" alt="Confusion Matrix - Adagrad">
  </div>
</div>

#### Stochastig Gradient Descent

<div class="grid-container">
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/predictions_SGD.png" alt="Predictions - SGD">
  </div>
  <div class="grid-item">
    <img src="https://rg625.github.io/assets/images/mnist_optimizer_comparison/confusion_matrix_SGD.png" alt="Confusion Matrix - SGD">
  </div>
</div>

## Sampling

The second application of the KW method involves estimating the score function of a probability distribution for use in two Markov Chain Monte Carlo (MCMC) sampling algorithms: Langevin Dynamics (LD) and Hamiltonian/Hybrid Monte Carlo (HMC). The performance of KW in this context was compared against PyTorch’s autodifferentiation-based approach and Spall’s SPSA method. For a comprehensive analysis, all these models were tested for different number of steps in the MCMC chains and samples. The main results displaying the evolution of a particular sample through the MCMC chain and the empirically found distribution are below, along with a comparison of memory usages and runtimes. More results are displayed on the project's page.

<div style="display: flex; flex-direction: column; gap: 20px;">
  <iframe src="https://rg625.github.io/assets/images/kw_sampling/convergence_plot_1_1000.html" width="100%" height="500px"></iframe>
  <iframe src="https://rg625.github.io/assets/images/kw_sampling/histogram_comparison_10000_1000.html" width="100%" height="500px"></iframe>
</div>

## Conclusion

The results indicate that different optimization algorithms exhibit varying levels of performance in terms of training loss, test loss, accuracy, computational cost, complexity, and training duration. The Adam optimizer consistently demonstrates strong performance, while other optimizers, such as SPSA and Kiefer-Wolfowitz, yield competitive results depending on the metric under consideration. The KW algorithm, however, is significantly slower than autodifferentiation-based optimizers, as it requires perturbing each parameter individually to estimate the gradient direction. This leads to a linear scaling in computational cost with respect to the number of parameters, making KW impractical for large-scale optimization tasks. While KW exhibits superior convergence properties compared to Adam on average, its computational inefficiency renders it infeasible for practical use. In contrast, SPSA provides a faster alternative at the cost of reduced accuracy.

Regarding the second application, the KW method demonstrates comparable performance to autodifferentiation in score estimation; however, it requires additional hyperparameter tuning, particularly in selecting the appropriate perturbation parameter.

The source code for these experiments is available on my [GitHub repository](https://github.com/rg625/KW_grad_estimation). Readers are encouraged to explore the implementation and reproduce the results.