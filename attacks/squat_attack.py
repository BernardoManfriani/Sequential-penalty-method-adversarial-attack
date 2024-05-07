import torch

# DA SOSTUTUIRE CON SQUAT
def fgsm_attack(image, epsilon, data_grad):
    # Calcolo il segno del gradiente rispetto all'immagine
    sign_data_grad = data_grad.sign()
    # Produco l'immagine perturbata aggiungendo l'epsilon moltiplicato per il segno del gradiente
    perturbed_image = image + epsilon*sign_data_grad
    # Clip dell'immagine perturbata in modo che sia compresa nell'intervallo [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Restituisco l'immagine perturbata
    return perturbed_image