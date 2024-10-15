import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from models.smallcnn import SmallCNN
import cvxpy as cp
import glob
from PIL import Image
import os
import re
import argparse

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_sorted_png_files(folder_path):
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    return sorted(png_files, key=natural_sort_key)

def create_gif_from_png(folder_path, output_path='output.gif', duration=500):
    """
    Create a GIF from PNG files in a specified folder.
    
    Parameters:
    folder_path (str): Path to the folder containing PNG files.
    output_path (str): Path where the output GIF will be saved. Default is 'output.gif'.
    duration (int): Duration of each frame in milliseconds. Default is 500ms.
    
    Returns:
    str: Path to the created GIF file.
    """
    # Get list of PNG files in the folder
    png_files = get_sorted_png_files(folder_path)
    
    if not png_files:
        raise ValueError(f"No PNG files found in {folder_path}")
    
    # Open all PNG files
    images = [Image.open(file) for file in png_files]
    
    # Save the first image as GIF and append the rest
    images[0].save(output_path, save_all=True, append_images=images[1:], 
                   duration=duration, loop=0)
    
    print(f"GIF created and saved as {output_path}")
    return output_path

def show(model, input_image, input_image_perturbed, perturbation, iterations, target_label, true_label):
    # Create a figure with four subplots in a single row
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
    ax3.set_title(f'Perturbation (target label: {target_label})')
    ax3.axis('off')
    fig.colorbar(perturbation_plot, ax=ax3, fraction=0.046, pad=0.04)
    
    # Logits
    with torch.no_grad():
        original_logits = torch.sigmoid(model(input_image).cpu().squeeze())
        perturbed_logits = torch.sigmoid(model(input_image_perturbed).cpu().squeeze()) 
    
    x = range(10)
    ax4.plot(x, original_logits, 'b-', label='Original')
    ax4.plot(x, perturbed_logits, 'r-', label='Perturbed')
    ax4.set_title('Model Logits')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Logit Value')
    ax4.legend()
    ax4.grid(True)
    
        
    os.makedirs(f"results/from_{true_label}_to_{target_label}", exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f"results/from_{true_label}_to_{target_label}/{true_label}_to_{target_label}_{iterations}.png")
    # plt.show()
    plt.close(fig)  

def SQUAT_attack(model, input_image, target_label, true_label, Niter=200, tau=0.5):
    print("=> Attacking the input image")
    perturbation = torch.zeros_like(input_image, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=0.01)
    
    for k in range(Niter):
        print(f"=> Solving the unconstraint subproblem {k+1}")
        
        optimizer.zero_grad()
        
        # Perturbed input
        input_image_perturbed = input_image + perturbation
        
        # f(xk) = 1/2 * ||x - xk||^2
        f = 0.5 * torch.norm(perturbation)**2
        
        # g(xk) = (IK - 1K^T*ej)C(xk)
        output = model(input_image_perturbed)
        IK = torch.eye(10, device=input_image.device)
        one_K = torch.ones(10, device=input_image.device)
        ej = torch.zeros(10, device=input_image.device)
        ej[target_label] = 1
        
        g = (IK - torch.outer(one_K, ej)) @ output.squeeze()
        
        # Loss function: f(xk) + tau * max{0, g(xk)}^2
        loss = f + tau * torch.sum(torch.relu(g)**2)
        
        loss.backward()
        optimizer.step()
        
        if k % 1 == 0:
            show(model, input_image, input_image_perturbed, perturbation, k, target_label.item(), true_label)

        # Check if the perturbation satisfies the misclassification constraint
        with torch.no_grad():
            output = model(input_image_perturbed)
            if output.argmax() == target_label:
                print("=> Model corrupted")
                break
        
        # Increase tau
        tau = tau * 1.5
    
    return input_image_perturbed, perturbation, k

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SQUAT Attack Script")
    parser.add_argument('--true-label', type=int, required=True, help="True label of the input image")
    parser.add_argument('--target-label', type=int, required=True, help="Target label for the attack")
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
    train_loader = DataLoader(d, batch_size=1, shuffle=True)

    # Get a single image and its true label
    data_iter = iter(train_loader)

    for input_image, true_label in data_iter:
        if true_label == args.true_label:
            true_label = torch.tensor(args.true_label)
            break
        
    # printa il type

    target_label = torch.tensor(args.target_label)

    # Move the image and label to the appropriate device (CPU or GPU)
    input_image = input_image.to(device)
    true_label = true_label.to(device)

    # Load the model (SmallCNN)
    model = SmallCNN().to(device)

    # Load the pre-trained model weights
    checkpoint = torch.load(f"checkpoints/smallcnn_mnist_best.pth", map_location=device, weights_only=True)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # Ensure the model is in evaluation mode
    model.eval()

    # Call the SQUAT attack (make sure it's implemented correctly)
    input_image_perturbed, perturbation, iterations = SQUAT_attack(model, input_image, target_label,true_label, Niter=100, tau=0.5, )

    show(model, input_image, input_image_perturbed, perturbation, iterations, target_label, true_label.item())
    plt.close('all')

    create_gif_from_png(f"results/from_{true_label.item()}_to_{target_label}", output_path=f"results/from_{true_label.item()}_to_{target_label}/output.gif", duration=500)

if __name__ == "__main__":
    main()