from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
import os
import re
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
    
    # Converti true_label in intero se è un tensore
    if isinstance(true_label, torch.Tensor):
        true_label_original = true_label.item()
    else:
        true_label_original = true_label
    
    # Original image (riporta i valori in [0, 1])
    original_image = input_image.detach().cpu().squeeze().permute(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    perturbed_image = input_image_perturbed.detach().cpu().squeeze().permute(1, 2, 0) 
    perturbed_image = (perturbed_image - perturbed_image.min()) / (perturbed_image.max() - perturbed_image.min())
    ax2.imshow(perturbed_image)
    ax2.set_title(f'Perturbed Image after {iterations} iterations')
    ax2.axis('off')

    #print perturbation
    # print(perturbation)
    #printa max e min e avg
    # print(f"Max: {perturbation.max().item()}, Min: {perturbation.min().item()}, Avg: {perturbation.mean().item()}")

    perturbation_plot = ax3.imshow(perturbation.detach().cpu().squeeze().permute(1, 2, 0), cmap='RdBu')
    ax3.set_title(f'Perturbation (target label: {target_label}, tau: {round(tau,2)}, rho: {rho})')
    ax3.axis('off')
    fig.colorbar(perturbation_plot, ax=ax3, fraction=0.046, pad=0.04)

    # Logits (visualizza le 10 classi con i punteggi più alti e aggiungi la classe target)
    with torch.no_grad():
        original_logits = torch.softmax(model(input_image), dim=1).cpu().squeeze()
        perturbed_logits = torch.softmax(model(input_image_perturbed), dim=1).cpu().squeeze()

    # Seleziona le prime 10 classi predette
    topk_original = torch.topk(original_logits, 10)
    topk_perturbed = torch.topk(perturbed_logits, 10)

    # Aggiungi la classe target se non è già tra le top 10
    target_value_original = original_logits[target_label].item()
    target_value_perturbed = perturbed_logits[target_label].item()
    
    indices = topk_original.indices.tolist()
    values_original = topk_original.values.tolist()
    values_perturbed = topk_perturbed.values.tolist()
    
    if target_label not in indices:
        indices.append(target_label)
        values_original.append(target_value_original)
        values_perturbed.append(target_value_perturbed)
    
    # Ordina per indici
    indices, values_original, values_perturbed = zip(*sorted(zip(indices, values_original, values_perturbed), key=lambda x: -x[1]))
    
    # Limita a massimo 11 classi, includendo la target
    indices = indices[:11]
    values_original = values_original[:11]
    values_perturbed = values_perturbed[:11]

    x = np.arange(len(indices))  # Indici per le classi
    bar_width = 0.35

    ax4.bar(x - bar_width / 2, values_original, bar_width, label='Original', color='b')
    ax4.bar(x + bar_width / 2, values_perturbed, bar_width, label='Perturbed', color='r')
    ax4.set_title(f'Model Logits after {iterations} iterations')
    ax4.set_xlabel('Classes')
    ax4.set_ylabel('Probability')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{i}' for i in indices], rotation=45, ha="right")
    ax4.legend()
    ax4.grid(True)

    # Save the figure
    os.makedirs(f"results/from_{true_label_original}_to_{target_label}", exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"results/from_{true_label_original}_to_{target_label}/{true_label_original}_to_{target_label}_{iterations}_tau_{round(tau, 2)}_rho_{rho}.png")
    plt.close(fig)


def spm_adv_attack(model, input_image, target_label, true_label, Niter, tau, rho):
    """Perform SPM adversarial attack."""
    print("=> Attacking the input image")
    perturbation = torch.zeros_like(input_image, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=0.001)
    # optimizer = torch.optim.Adam([perturbation], lr=0.02)

    print(f"=> Using Adam optimizer")
    
    for k in range(Niter):
        print(f"=> Solving the unconstraint subproblem {k + 1}")
        
        optimizer.zero_grad()
        
        # Perturbed input
        input_image_perturbed = input_image + perturbation
        
        # perturbation_loss = 1/2 * ||x - xk||^2
        perturbation_loss = 0.5 * torch.norm(perturbation) ** 2
        
        with open("data/img/imagenet_classes.txt") as class_file:
            labels = [line.strip() for line in class_file.readlines()]

        # with torch.no_grad():
        #     output = model(input_image_perturbed)
        #     _, predicted_class = output.max(1)
        #     class_name = labels[predicted_class.item()]
        #     print(f"Classe predetta: {class_name}")

        output = model(input_image_perturbed)
        # g(xk) = (IK - 1K^T*ej)C(xk)
        num_classes = model.fc.out_features  # Get the number of classes from the model's final layer
        IK = torch.eye(num_classes, device=input_image.device)
        one_K = torch.ones(num_classes, device=input_image.device)
        ej = torch.zeros(num_classes, device=input_image.device)
        ej[target_label] = 1
        
        g = (IK - torch.outer(one_K, ej)) @ output.squeeze()
        
        # Loss function: perturbation_loss + tau * max{0, g(xk)}^2
        loss = perturbation_loss + tau * torch.sum(torch.relu(g) ** 2)   
        loss.backward()
        optimizer.step()
        
        if k % 1 == 0:
            show(model, input_image, input_image_perturbed, perturbation, k, target_label, true_label, tau, rho)

        # Check if the perturbation satisfies the misclassification constraint
        with torch.no_grad():
            output = model(input_image_perturbed)
            if output.argmax() == target_label:
                print("=> Model corrupted")
                print (f"Classe predetta: {labels[output.argmax().item()]}")
                break
        
        # Increase tau
        tau = tau * rho
    
    return input_image_perturbed, perturbation, k


# Main function
def main():
    # Seleziona il dispositivo (GPU se disponibile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crea il modello e carica i pesi preaddestrati
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)

    # Congela i parametri se non vuoi riaddestrare il modello
    for param in model.parameters():
        param.requires_grad = False

    # Definisci le trasformazioni per le immagini di ImageNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carica un'immagine a caso da ImageNet
    # per ogni file nella cartella data/img/imagenet-sample-images-master seleziona un'immagine a caso
    image_path = np.random.choice(glob.glob("data/img/imagenet-sample-images-master/*"))

    # image_path = 'data/img/imagenet-sample-images-master/image_test3.JPEG'  # Sostituisci con il percorso della tua immagine
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Aggiungi una dimensione batch

    # Carica i nomi delle classi di ImageNet
    with open("data/img/imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # Metti il modello in modalità di valutazione
    model.eval()

    # Effettua una predizione sull'immagine
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = output.max(1)
        class_name = labels[predicted_class.item()]
        print(f"Classe predetta: {class_name}")

    # Definisci la classe target per l'attacco
    target_label = 71  # Cambia con la tua classe target
    print(f"Classe target: {labels[target_label]}")

    # Definisci i parametri per l'attacco
    tau = 1
    rho = 1.5
    Niter = 1000

    # Esegui l'attacco SPM 
    input_perturbed, perturbation, iterations = spm_adv_attack(model, input_tensor, target_label, predicted_class, Niter, tau, rho)

    # Effettua una predizione sull'immagine perturbata
    with torch.no_grad():
        output_perturbed = model(input_perturbed)
        _, predicted_class_perturbed = output_perturbed.max(1)
        class_name_perturbed = labels[predicted_class_perturbed.item()]
        print(f"Classe predetta (perturbata): {class_name_perturbed}")
    
    # Mostra e salva i risultati
    show(model, input_tensor, input_perturbed, perturbation, iterations, target_label, predicted_class.item(), tau, rho)

    # Crea un GIF dell'attacco
    create_gif_from_png(f"results/from_{predicted_class.item()}_to_{target_label}", 
                        output_path=f"results/from_{predicted_class.item()}_to_{target_label}/output.gif", 
                        duration=500)
if __name__ == '__main__':
    main()
