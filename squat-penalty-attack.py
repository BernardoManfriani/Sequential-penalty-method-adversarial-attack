import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from models.smallcnn import SmallCNN
import glob
from PIL import Image
import os
import re
import argparse
import numpy as np

def natural_sort_key(s):
    """Helper function to sort file names numerically."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_sorted_png_files(folder_path):
    """Get all PNG files from a folder, sorted naturally by file name."""
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    return sorted(png_files, key=natural_sort_key)

def create_gif_from_png(folder_path, output_path='output.gif', duration=500):
    """Create a GIF from PNG images in the given folder."""
    png_files = get_sorted_png_files(folder_path)
    
    if not png_files:
        raise ValueError(f"No PNG files found in {folder_path}")
    
    images = [Image.open(file) for file in png_files]
    
    images[0].save(output_path, save_all=True, append_images=images[1:], 
                   duration=duration, loop=0)
    
    print(f"GIF created and saved as {output_path}")
    return output_path

def show(model, input_image, input_image_perturbed, perturbation, iterations, target_label, true_label, tau, rho):
    """Plot and save the original image, perturbed image, perturbation, and logits."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    ax1.imshow(input_image.detach().cpu().squeeze(), cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Perturbed image
    ax2.imshow(input_image_perturbed.detach().cpu().squeeze(), cmap='gray')
    ax2.set_title(f'Perturbed Image after {iterations} iterations')
    ax2.axis('off')

    # Perturbation
    perturbation_plot = ax3.imshow(perturbation.detach().cpu().squeeze(), cmap='RdBu', norm=plt.Normalize(vmin=-0.5, vmax=0.5))
    ax3.set_title(f'Perturbation (target label: {target_label}, tau: {round(tau,2)})')
    ax3.axis('off')
    fig.colorbar(perturbation_plot, ax=ax3, fraction=0.046, pad=0.04)

    # Logits
    with torch.no_grad():
        original_logits = torch.sigmoid(model(input_image).cpu().squeeze())
        perturbed_logits = torch.sigmoid(model(input_image_perturbed).cpu().squeeze())

    x = np.arange(10)
    bar_width = 0.35

    ax4.bar(x - bar_width / 2, original_logits, bar_width, label='Original', color='b')
    ax4.bar(x + bar_width / 2, perturbed_logits, bar_width, label='Perturbed', color='r')
    ax4.set_title(f'Model Logits after {iterations} iterations')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Logit Value')
    ax4.legend()
    ax4.grid(True)

    # Save the figure
    os.makedirs(f"results/from_{true_label}_to_{target_label}", exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"results/from_{true_label}_to_{target_label}/{true_label}_to_{target_label}_{iterations}_tau_{round(tau, 2)}_rho_{rho}.png")
    plt.close(fig)

def SQUAT_attack(model, input_image, target_label, true_label, Niter, tau, rho):
    """Perform SQUAT adversarial attack."""
    print("=> Attacking the input image")
    perturbation = torch.zeros_like(input_image, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=0.01)
    
    for k in range(Niter):
        print(f"=> Solving the unconstraint subproblem {k + 1}")
        
        optimizer.zero_grad()
        
        # Perturbed input
        input_image_perturbed = input_image + perturbation
        
        # f(xk) = 1/2 * ||x - xk||^2
        f = 0.5 * torch.norm(perturbation) ** 2
        
        # g(xk) = (IK - 1K^T*ej)C(xk)
        output = model(input_image_perturbed)
        IK = torch.eye(10, device=input_image.device)
        one_K = torch.ones(10, device=input_image.device)
        ej = torch.zeros(10, device=input_image.device)
        ej[target_label] = 1
        
        g = (IK - torch.outer(one_K, ej)) @ output.squeeze()
        
        # Loss function: f(xk) + tau * max{0, g(xk)}^2
        loss = f + tau * torch.sum(torch.relu(g) ** 2)
        
        loss.backward()
        optimizer.step()
        
        if k % 1 == 0:
            show(model, input_image, input_image_perturbed, perturbation, k, target_label.item(), true_label.item(), tau, rho)

        # Check if the perturbation satisfies the misclassification constraint
        with torch.no_grad():
            output = model(input_image_perturbed)
            if output.argmax() == target_label:
                print("=> Model corrupted")
                break
        
        # Increase tau
        tau = tau * rho
    
    return input_image_perturbed, perturbation, k

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SQUAT Attack Script")
    parser.add_argument('--true-label', type=int, required=True, help="True label of the input image")
    parser.add_argument('--target-label', type=int, required=True, help="Target label for the attack")
    parser.add_argument('--tau', type=float, default=1, help="Penalty parameter")
    parser.add_argument('--rho', type=float, default=1.5, help="Incremental coefficient for the penalty parameter")
    parser.add_argument('--Niter', type=int, default=1000, help="Number of iterations")
    args = parser.parse_args()

    # Select the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    # Define MNIST data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST stats
    ])

    # Load MNIST dataset
    d = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(d, batch_size=1, shuffle=False)

    # Get a single image and its true label
    data_iter = iter(train_loader)

    for input_image, true_label in data_iter:
        if true_label == args.true_label:
            true_label = torch.tensor(args.true_label)
            break
        
    target_label = torch.tensor(args.target_label)

    # Move the image and label to the appropriate device (CPU or GPU)
    input_image = input_image.to(device)
    true_label = true_label.to(device)

    # Load the model (SmallCNN)
    model = SmallCNN().to(device)

    # Load the pre-trained model weights
    checkpoint = torch.load(f"checkpoints/smallcnn_mnist_best.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    # Ensure the model is in evaluation mode
    model.eval()

    # Perform the SQUAT attack
    input_image_perturbed, perturbation, iterations = SQUAT_attack(
        model, input_image, target_label, true_label, args.Niter, args.tau, args.rho)

    # Show and save the final result
    show(model, input_image, input_image_perturbed, perturbation, iterations, target_label.item(), true_label.item(), args.tau, args.rho)
    
    # Create a GIF of the attack
    create_gif_from_png(f"results/from_{true_label.item()}_to_{target_label.item()}", 
                        output_path=f"results/from_{true_label.item()}_to_{target_label.item()}/output.gif", 
                        duration=500)

if __name__ == "__main__":
    main()
