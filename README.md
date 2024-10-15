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
python squat-penalty-attack.py  --true-label 5 --target-label 1
```

The results are saved in the `results/` directory.

### Results

## References

- Beuzeville et al., "Adversarial Attacks via Sequential Quadratic Programming", 2024. \cite{beuzeville2024}
- Rony et al., "Sequential Penalty Method for Adversarial Attacks", 2021. \cite{rony2021}
- Grippo et al., "Sequential Penalty Methods: Theory and Applications", 2023. \cite{grippo2023}


