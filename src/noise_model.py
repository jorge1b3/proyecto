"""
Modelo de ruido Poisson usando DeepInverse
"""

import torch
import numpy as np
import deepinv as dinv
from config import GAMMA, DEVICE

def create_poisson_physics(gamma: float = GAMMA) -> dinv.physics.Denoising:
    """
    Crear el modelo de física de ruido Poisson
    
    Args:
        gamma: Parámetro de ganancia/exposición
        
    Returns:
        Modelo de física de DeepInverse
    """
    # Crear el modelo de ruido Poisson 
    # normalize=False para manejar manualmente la escala
    noise_model = dinv.physics.PoissonNoise(
        gain=gamma, 
        normalize=True,  # Manejaremos la normalización manualmente
        clip_positive=False  # Asegurar valores positivos
    )
    
    # Crear el physics model usando DeepInverse
    physics = dinv.physics.Denoising(noise_model)
    
    return physics

def apply_poisson_noise(image: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    Aplicar ruido Poisson a una imagen
    
    Args:
        image: Imagen limpia en rango [0,1]
        gamma: Parámetro de ganancia (nivel de ruido inverso - más pequeño = más ruido)
        
    Returns:
        Imagen con ruido Poisson en rango [0,1]
    """

    physics = create_poisson_physics(gamma)
    
    x_scaled = image
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    print(f"Imagen escalada para Poisson - min: {x_tensor.min():.4f}, max: {x_tensor.max():.4f}")
    
    with torch.no_grad():
        y_tensor = physics(x_tensor)
    
    y_clipped = torch.clamp(y_tensor, 0.0, 1.0).squeeze()
    
    # Convertir a numpy
    y = y_clipped.cpu().numpy()
    
    print(f"Imagen con ruido final - min: {y.min():.4f}, max: {y.max():.4f}")
    print(f"Porcentaje de pixeles en [0, 0.1]: {np.mean(y <= 0.1)*100:.1f}%")
    print(f"Porcentaje de pixeles en [0.9, 1.0]: {np.mean(y >= 0.9)*100:.1f}%")
    
    return y
