## Overview

This repository contains the implementation of adversarial attacks on the MNIST dataset using.
  

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

## Sequential Penalty Methods
The Sequential Penalty Method consists of starting from the constrained problem:

$$\min f(x) \quad \text{s.t.} \quad g(x) \leq 0,$$

and solving a sequence of unconstrained subproblems of the form:

$$F(x) = f(x) + \tau \cdot P(x)$$
where $\tau$ is a penalty parameter that is increased at each iteration and $P(x) = \max \\{0, g(x)\\}^2$.

As shown in the figure below, the higher the value of $\tau$, the greater the tendency of the solution of $F(x)$ to coincide with the solution of the original problem $f(x)$.
<p align="center">
  <img src="https://github.com/user-attachments/assets/4fe9ed38-2760-46b2-b1d9-d077e4e5a766" width="500" alt="From 2 to 6, tau0=1, rho=1.2" title="From 2 to 6, tau0=1, rho=1.5"/> 
</p> 

In our specific case we have:
- **Target image**: $x$ 
- **Image to perturb**: $x_k$
- **Objective function**: $f(x_k) = \frac{1}{2} ||x-x_k||^2$
- **Constraint function**: $g(x_k)= (I_k - 1_K^T\cdot ej )\cdot C(x_k)$

We solve the original constrained problem:

$$\min \frac{1}{2} ||x-x_k||^2 \quad s.t. \quad (I_k - 1_K^T\cdot ej )\cdot C(x_k) \leq 0$$

by solving unconstrained subproblems using a Sequential Penalty approach:

$$
\begin{cases}
F(x_k) = f(x_k) + \tau \cdot P(x_k) \qquad \tau \geq 1 \\ \\
P(x_k)=  \max\\{0,g(x_k)\\}^2
\end{cases}
$$
    
The unconstrained subproblem to minimize becomes: 

$$\min_{x_k} \frac{1}{2} ||x-x_k||^2 + \tau \cdot \max\{0, (I_k - 1_K^T\cdot ej )\cdot C(x_k)\}^2$$

## Experiments
Several experiments were conducted to assess the effectiveness of the Sequential Penalty Method under different conditions:

- **Different Images**: We also experimented with attacking different images from the MNIST dataset, varying the true label and target label combinations. The attack was consistently effective across different samples, although certain digit pairs required more iterations due to visual similarities between digits.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb343c79-1863-4e75-94b5-cc97d27046e01" width="800" alt="From 4 to 9" title="From 4 to 9"/> 
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/351b730e-c42c-4fc2-a91e-201fc00a5d2f" width="800" alt="From 3 to 0" title="From 3 to 0"/> 
</p> 

- **Tau Values**: We tested different values for the hyperparameter tau to analyze its impact on the perturbation magnitude and misclassification rate. The results showed that smaller tau values resulted in smaller perturbations but required more iterations to achieve the target misclassification.
<p align="center">
  <img src="https://github.com/user-attachments/assets/cbe81d1b-995d-4142-a349-5c097b97d6aa" width="800" alt="From 2 to 6, tau0=1, rho=1.2" title="From 2 to 6, tau0=1, rho=1.2"/> 
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/cbe81d1b-995d-4142-a349-5c097b97d6aa" width="800" alt="From 2 to 6, tau0=1, rho=1.2" title="From 2 to 6, tau0=1, rho=1.5"/> 
</p> 

## Usage

### Training the Model

To train the SmallCNN model on the MNIST dataset, use the following command:

```bash
python main.py --epochs 10
```

This will train the model for 10 epochs and save the weights in the `checkpoints/` directory.

### Running the Adversarial Attacks

To run the adversarial attacks using the Sequential Penalty Method, execute the following command:

```bash
python squat-penalty-attack.py --true-label 3 --target-label 7 --tau 1.0 --rho 1.5 --Niter 1000
```

The results are saved in the `results/` directory.


## References

- Beuzeville et al., "Adversarial Attacks via Sequential Quadratic Programming", 2024. \cite{beuzeville2024}
- Rony et al., "Sequential Penalty Method for Adversarial Attacks", 2021. \cite{rony2021}
- Grippo et al., "Sequential Penalty Methods: Theory and Applications", 2023. \cite{grippo2023}


