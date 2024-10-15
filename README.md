# Sequential Penalty Method For Adversarial Attack on MNIST

## Overview

This repository contains the implementation of adversarial attacks on the MNIST dataset using...
  

## Introduction

Adversarial attacks are techniques used to deceive machine learning models by introducing small, carefully crafted perturbations to the input data. These perturbations are generally imperceptible to humans but can cause a model to make incorrect predictions...….

## Requirements

To run this project, you will need the following dependencies:

- cose
- cose

You can install all the required libraries using:

```bash
pip install -r requirements.txt
```

## Model and Dataset

The adversarial attacks are performed on images from the MNIST dataset. The model used is a Small Convolutional Neural Network (SmallCNN) trained for 20 epochs, achieving 99% accuracy on non-perturbed images. The architecture and training details can be found in the `models/smallcnn.py` file.

## Adversarial Attack Algorithm

### Datails

## Experiments
Several experiments were conducted to assess the effectiveness of the Sequential Penalty Method under different conditions:

- **Different Images**: We also experimented with attacking different images from the MNIST dataset, varying the true label and target label combinations. The attack was consistently effective across different samples, although certain digit pairs required more iterations due to visual similarities between digits.

<p align="center">
  <img src="https://github.com/user-attachments/assets/888880bc-edac-4a73-b4e9-7c29c488e201" width="800" alt="From 1 to 5" title="From 1 to 5"/> 
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/67cfbc8f-b327-43f8-9ec4-4f5fa2d7b3b3" width="800" alt="From 8 to 0" title="From 8 to 0"/> 
</p> 

- **Tau Values**: We tested different values for the hyperparameter tau to analyze its impact on the perturbation magnitude and misclassification rate. The results showed that smaller tau values resulted in smaller perturbations but required more iterations to achieve the target misclassification.
<p align="center">
  <img src="https://github.com/user-attachments/assets/0adac4ba-4fb3-4444-8726-04e424789d85" width="800" alt="From 4 to 9" title="From 4 to 9"/> 
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/67cfbc8f-b327-43f8-9ec4-4f5fa2d7b3b3" width="800" alt="From 8 to 0" title="From 8 to 0"/> 
</p> 

## Usage

### Training the Model

To train the SmallCNN model on the MNIST dataset, use the following command:

```bash
python main.py --epochs 10
```

This will train the model for 10 epochs and save the weights in `checkpoints/` directory.

### Running Adversarial Attacks

To run adversarial attacks using the Sequential Penalty Method, execute the following command:

```bash
python squat-penalty-attack.py  --true-label 3 --target-label 7 --tau 1.0 --rho 1.5 --Niter 1000
```

The results are saved in the `results/` directory.


### Results

## References

- Beuzeville et al., "Adversarial Attacks via Sequential Quadratic Programming", 2024. \cite{beuzeville2024}
- Rony et al., "Sequential Penalty Method for Adversarial Attacks", 2021. \cite{rony2021}
- Grippo et al., "Sequential Penalty Methods: Theory and Applications", 2023. \cite{grippo2023}


