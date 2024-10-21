import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import glob
from PIL import Image
import os
import re
import argparse
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

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

def denormalize(tensor):
    """Denormalize the tensor using ImageNet mean and std."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def show(model, input_image, input_image_perturbed, perturbation, iterations, target_label, true_label, tau, rho):
    """Plot and save the original image, perturbed image, perturbation, and logits."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    original = denormalize(input_image.detach().cpu().squeeze())
    ax1.imshow(original.permute(1, 2, 0).clip(0, 1))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Perturbed image
    perturbed = denormalize(input_image_perturbed.detach().cpu().squeeze())
    ax2.imshow(perturbed.permute(1, 2, 0).clip(0, 1))
    ax2.set_title(f'Perturbed Image after {iterations} iterations')
    ax2.axis('off')

    # Perturbation
    perturbation_display = perturbation.detach().cpu().squeeze().permute(1, 2, 0)
    perturbation_display = (perturbation_display - perturbation_display.min()) / (perturbation_display.max() - perturbation_display.min())
    perturbation_plot = ax3.imshow(perturbation_display, cmap='RdBu')
    ax3.set_title(f'Perturbation (target label: {target_label}, tau: {round(tau,2)}, rho: {rho})')
    ax3.axis('off')
    fig.colorbar(perturbation_plot, ax=ax3, fraction=0.046, pad=0.04)


    # Logits    
    with torch.no_grad():
        original_logits = torch.softmax(model(input_image).cpu().squeeze(), dim=0)
        perturbed_logits = torch.softmax(model(input_image_perturbed).cpu().squeeze(), dim=0)

    # Seleziona le prime 10 classi predette
    topk_original = torch.topk(original_logits, 10)
    topk_perturbed = torch.topk(perturbed_logits, 10)

    # Ottieni gli indici e i valori delle top 10 classi
    indices_original = topk_original.indices.tolist()
    values_original = topk_original.values.tolist()
    values_perturbed = topk_perturbed.values.tolist()

    # Controlla se la classe target è tra le top 10
    if target_label not in indices_original:
        # Se non è presente, aggiungila alla lista e includi i suoi valori predetti
        indices_original.append(target_label)
        values_original.append(original_logits[target_label].item())
        values_perturbed.append(perturbed_logits[target_label].item())

    # Crea un dizionario per accoppiare indice e valore e ordina
    paired = list(zip(indices_original, values_original, values_perturbed))
    # Ordina per probabilità del modello originale in ordine decrescente, assicurando che la classe target sia inclusa
    paired = sorted(paired, key=lambda x: (x[0] == target_label, x[1]), reverse=True)

    # Mantieni solo i primi 10, assicurando che la classe target sia inclusa
    if len(paired) > 10:
        # Se la classe target è nell'elenco ma non è nei primi 10, sostituisci l'ultimo con la classe target
        if target_label not in [x[0] for x in paired[:10]]:
            paired[-1] = next(item for item in paired if item[0] == target_label)

    # Prendi solo le prime 10 classi
    paired = paired[:10]

    # Separa di nuovo gli indici e i valori
    indices, values_original, values_perturbed = zip(*paired)

    # Plot dei risultati
    x = np.arange(len(indices))
    bar_width = 0.35

    ax4.bar(x - bar_width / 2, values_original, bar_width, label='Original', color='b')
    ax4.bar(x + bar_width / 2, values_perturbed, bar_width, label='Perturbed', color='r')
    ax4.set_title(f'Model Logits after {iterations} iterations')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Probability')
    ax4.set_xticks(x)
    ax4.set_xticklabels(indices, rotation=45, ha='right') # Imposta le etichette delle classi
    ax4.legend()
    ax4.grid(True)

    # Save the figure
    os.makedirs(f"results/imagenet/from_{true_label}_to_{target_label}", exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"results/imagenet/from_{true_label}_to_{target_label}/{true_label}_to_{target_label}_{iterations}_tau_{round(tau, 2)}_rho_{rho}.png")
    plt.close(fig)

def spm_adv_attack_imagenet(model, input_image, target_label, true_label, Niter, tau, rho):

    """Perform Sequential Penalty Attack adversarial attack."""
    print("=> Attacking the input image")
    perturbation = torch.zeros_like(input_image, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=0.001)
    # optimizer = torch.optim.SGD([perturbation], lr=0.01)
    print(f"=> Using Adam optimizer")
    
    for k in range(Niter):
        print(f"=> Solving the unconstraint subproblem {k + 1}")
        
        optimizer.zero_grad()
        
        # Perturbed input
        input_image_perturbed = input_image + perturbation
        
        # f(xk) = 1/2 * ||x - xk||^2
        f = 0.5 * torch.norm(perturbation) ** 2
        
        # g(xk) = (IK - 1K^T*ej)C(xk)
        K = 1000
        output = model(input_image_perturbed)
        IK = torch.eye(K, device=input_image.device)
        one_K = torch.ones(K, device=input_image.device)
        ej = torch.zeros(K, device=input_image.device)
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
                print(f"=> Model corrupted (predicted class: {output.argmax()})")
                break
        
        # Increase tau
        tau = tau * rho
    
    return input_image_perturbed, perturbation, k

def load_and_preprocess_image(image_path, device):
    """Load and preprocess the image for the ResNet18 model."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def main():
     # Parse command line arguments
    parser = argparse.ArgumentParser(description="SPM Attack Script")
    parser.add_argument('--tau', type=float, default=1, help="Penalty parameter")
    parser.add_argument('--rho', type=float, default=1.5, help="Incremental coefficient for the penalty parameter")
    parser.add_argument('--Niter', type=int, default=100, help="Number of iterations")
    parser.add_argument('--target-label', type=int, required=True, help="The target class label for the adversarial attack")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the input image file")
    args = parser.parse_args()

    # Select the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    # Load and preprocess the image
    input_image = load_and_preprocess_image(args.image_path, device)
    
    target_label = torch.tensor(args.target_label).to(device)

    # Load the model (ResNet18)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    true_label = model(input_image).argmax()
    print(f"First model prediction: {model(input_image).argmax().item()}")
    #print target class object from imagenet_classes.txt
    with open('data/img/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"True label: {classes[target_label.item()]}")

    # Perform the SPM attack
    input_image_perturbed, perturbation, iterations = spm_adv_attack_imagenet(
        model, input_image, target_label, true_label, args.Niter, args.tau, args.rho)

    # Save the tensor
    torch.save(input_image_perturbed, "results/imagenet/perturbed_image_tensor.pt")

    # Show and save the final result
    show(model, input_image, input_image_perturbed, perturbation, iterations, target_label.item(), true_label.item(), args.tau, args.rho)

    # Create a GIF of the attack
    create_gif_from_png(f"results/imagenet/from_{true_label.item()}_to_{target_label.item()}",
                        output_path=f"results/imagenet/from_{true_label.item()}_to_{target_label.item()}/output.gif",
                        duration=500)


if __name__ == "__main__":
    main()


