---
layout: post
title: "MNIST Optimizer Comparison"
date: 2025-02-15 16:12:06
author: rg625
---

## Introduction

In this blog post, we will compare different optimization algorithms for training a Convolutional Neural Network (CNN) on the MNIST dataset. The MNIST dataset consists of handwritten digits and is a common benchmark for evaluating machine learning models.

## Methodology

We trained a simple CNN model using various optimizers on the MNIST dataset. The optimizers used in this study are:

- SPSAOptimizerSimple
- SPSAOptimizerAdaptive
- KieferWolfowitzSimple
- KieferWolfowitzAdaptive
- Adam
- Adagrad
- SGD

The model architecture is as follows:

- Two convolutional layers with ReLU activation and max pooling.
- Two fully connected layers with ReLU activation.

We trained the model for 10 epochs with a batch size of 1024 and a learning rate of 0.01. The performance of each optimizer was evaluated based on training loss, test loss, training accuracy, test accuracy, compute cost, complexity, and training time.

## Results

### Training and Test Loss

![Train Loss](optimizer_plots/mnist_classifier/comparative/comparative_train_loss.png)
![Test Loss](optimizer_plots/mnist_classifier/comparative/comparative_test_loss.png)

### Training and Test Accuracy

![Train Accuracy](optimizer_plots/mnist_classifier/comparative/comparative_train_accuracy.png)
![Test Accuracy](optimizer_plots/mnist_classifier/comparative/comparative_test_accuracy.png)

### Compute Cost and Complexity

![Compute Cost](optimizer_plots/mnist_classifier/comparative/comparative_compute_cost.png)
![Complexity](optimizer_plots/mnist_classifier/comparative/comparative_complexity.png)

### Training Time

![Time](optimizer_plots/mnist_classifier/comparative/comparative_time.png)

### Predictions and Confusion Matrices

Below are the predictions and confusion matrices for each optimizer. Each figure shows a 4x4 grid of randomly sampled digits from the test set, along with the true and predicted labels.

#### SPSAOptimizerSimple

![Predictions](optimizer_plots/mnist_classifier/predictions_SPSAOptimizerSimple.png)
![Confusion Matrix](optimizer_plots/mnist_classifier/confusion_matrix_SPSAOptimizerSimple.png)

#### SPSAOptimizerAdaptive

![Predictions](optimizer_plots/mnist_classifier/predictions_SPSAOptimizerAdaptive.png)
![Confusion Matrix](optimizer_plots/mnist_classifier/confusion_matrix_SPSAOptimizerAdaptive.png)

#### KieferWolfowitzSimple

![Predictions](optimizer_plots/mnist_classifier/predictions_KieferWolfowitzSimple.png)
![Confusion Matrix](optimizer_plots/mnist_classifier/confusion_matrix_KieferWolfowitzSimple.png)

#### KieferWolfowitzAdaptive

![Predictions](optimizer_plots/mnist_classifier/predictions_KieferWolfowitzAdaptive.png)
![Confusion Matrix](optimizer_plots/mnist_classifier/confusion_matrix_KieferWolfowitzAdaptive.png)

#### Adam

![Predictions](optimizer_plots/mnist_classifier/predictions_Adam.png)
![Confusion Matrix](optimizer_plots/mnist_classifier/confusion_matrix_Adam.png)

#### Adagrad

![Predictions](optimizer_plots/mnist_classifier/predictions_Adagrad.png)
![Confusion Matrix](optimizer_plots/mnist_classifier/confusion_matrix_Adagrad.png)

#### SGD

![Predictions](optimizer_plots/mnist_classifier/predictions_SGD.png)
![Confusion Matrix](optimizer_plots/mnist_classifier/confusion_matrix_SGD.png)

## Conclusion

From the results, we can observe that different optimizers have varying performance in terms of training loss, test loss, accuracy, compute cost, complexity, and training time. Adam optimizer consistently shows strong performance, while other optimizers like SPSA and Kiefer-Wolfowitz also show competitive results depending on the metric being considered.

This study provides insights into the trade-offs involved in selecting an optimizer for training neural networks. Depending on the specific requirements of the application, one can choose an optimizer that balances accuracy, computational efficiency, and memory usage.

Feel free to explore the code and reproduce the results. The code is available on my [GitHub repository](https://github.com/rg625/mnist-optimizer-comparison).

Happy experimenting!

