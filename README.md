## Overview

This repository contains the implementation of adversarial attacks on the MNIST dataset using the Sequential Penalty Method (SPM). The approach focuses on generating adversarial examples that force a classifier to misclassify an image by introducing minimal perturbations, ensuring the changes are imperceptible to humans while still deceiving the machine learning model.

## Introduction

Adversarial attacks are techniques used to deceive machine learning models by introducing small, carefully crafted perturbations to the input data. These perturbations are generally imperceptible to humans but can cause a model to make incorrect predictions. In this project, we implement a **Sequential Penalty Method** to carry out adversarial attacks on images from the MNIST dataset.

The **Sequential Penalty Method (SPM)** transforms a constrained optimization problem into a series of unconstrained subproblems by introducing a penalty function. At each iteration, the penalty parameter is increased to enforce constraint satisfaction while minimizing the objective function. This method ensures that the adversarial perturbation remains small while still achieving misclassification.

### Problem Formulation

The constrained optimization problem is defined as follows:


$$\min f(x) \quad \text{s.t.} \quad g(x) \leq 0 $$

Where:

- $ f(x) = \frac{1}{2} \|x - x_k\|^2$ : This objective function minimizes the difference between the perturbed image $x$ and the original image $x_k$.

- $g(x) = (I_k - 1^T_k \cdot e_j) \cdot C(x_k) $: This inequality constraint ensures that the classifier misclassifies the perturbed image into the target class $j $. Here, $C(x_k) $ represents the classifier's prediction output, and $e_j $ is the target class vector.

The penalty function $P(x) = \max(0, g(x))^2 $ is applied iteratively, and the penalty parameter $\tau $ is increased over time to force the solution towards satisfying the constraints while minimizing $f(x) $.



## Requirements

Requirements

To run this project, you will need to install all the requirements with following command.

```bash
pip install -r requirements.txt
```

## Model and Dataset

The adversarial attacks are performed on images from the MNIST dataset. The model used is a Small Convolutional Neural Network (SmallCNN) trained for 10 epochs, achieving 99% accuracy on non-perturbed images. The architecture and training details can be found in the models/smallcnn.py file.
<p align="center">
  <img src="https://github.com/user-attachments/assets/203e432f-21fe-4cb5-9728-374c5338b3cf" width="300" alt="SmallCNN" title="SmallCNN"/> 
</p> 

The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels, split into 60,000 training images and 10,000 test images.

The SmallCNN model consists of two convolutional layers followed by ReLU activations, then a max pooling layer, followed by two more convolutional layers with ReLU activations, another max pooling layer, and finally three fully connected layers for classification.

The model was trained using the Adam optimizer (learning rate of 0.001) with cross-entropy loss, achieving high accuracy on non-perturbed images.
<p align="center">
  <img src="https://github.com/user-attachments/assets/c9f760b9-4aab-45e4-96c0-6deed39e8961" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/cc7cafaf-6cf1-4f4f-86b8-234e933faae4" width="400" alt="Train Accuracy" title="Train Accuracy"/>
</p> 

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

- **Tau Values**: We tested different values for the hyperparameter tau to analyze its impact on the perturbation magnitude and misclassification rate. 
<p align="center">
  <img src="https://github.com/user-attachments/assets/c8987a5e-6e5e-4347-8aa3-6e88abe2db76" width="800" alt="From 8 to 1, tau0=1, rho=1.1" title="From 8 to 1, tau0=1, rho=1.2"/> 
</p> 
<p align="center">
  <img src="https://github.com/user-attachments/assets/c8987a5e-6e5e-4347-8aa3-6e88abe2db76" width="800" alt="From 8 to 1, tau0=1, rho=1.5" title="From 8 to 1, tau0=1, rho=1.2"/> 
</p> 

| tau  | rho  | iterations |
|------|------|------------|
| 1.0  | 1.1  | 51       |
| 1.0  | 1.5  | 50        |

AGGIUNGI COMMENTO

- **Different Learning Rates**:  We tested different values for the learning rate of the Adam optimizer.
 * **lr=0.1**
  <p align="center">
  <img src="https://github.com/user-attachments/assets/f402117f-3441-4310-a5bf-1c1b94d5875f" width="800" alt="From 8 to 1, tau0=1, rho=1.1" title="From 8 to 1, tau0=1, rho=1.5, lr=0.1"/> 
</p> 

 * **lr=0.01**
<p align="center">
<img src="https://github.com/user-attachments/assets/ecf9fcbc-a102-4ce9-8c75-7e539cefb623" width="800" alt="From 8 to 1, tau0=1, rho=1.5" title="From 8 to 1, tau0=1, rho=1.5, lr=0.01"/> 
</p> 

| tau  | rho  | lr | iterations |
|------|------|----|------------|
| 1.0  | 1.5  | 0.1 |      9      |
| 1.0  | 1.5  | 0.01 |      52      |

- **Different Optimizer**: we compared Adam and SGD:
* **ADAM**
<p align="center">
<img src="https://github.com/user-attachments/assets/ecf9fcbc-a102-4ce9-8c75-7e539cefb623" width="800" alt="From 8 to 1, tau0=1, rho=1.5" title="From 8 to 1, tau0=1, rho=1.5, lr=0.01"/> 
</p> 

  
* **SGD**
<p align="center">
<img src="https://github.com/user-attachments/assets/86fe880d-21de-4d9a-863b-d5ab8644f06c" width="800" alt="From 8 to 1, SGD, lr=0.01" title="From 8 to 1, SGD, lr=0.01"/> 
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


