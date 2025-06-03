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
        normalize=False,  # Manejaremos la normalización manualmente
        clip_positive=True  # Asegurar valores positivos
    )
    
    # Crear el physics model usando DeepInverse
    physics = dinv.physics.Denoising(noise_model)
    
    return physics

def apply_poisson_noise(image: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    Aplicar ruido Poisson a una imagen
    
    Args:
        image: Imagen limpia en rango [0,1]
        gamma: Parámetro de ganancia
        
    Returns:
        Imagen con ruido Poisson en rango [0,1]
    """
    physics = create_poisson_physics(gamma)
    
    # Escalar la imagen para que funcione bien con Poisson
    x_scaled = image * gamma
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    print(f"Imagen escalada para Poisson - min: {x_tensor.min():.4f}, max: {x_tensor.max():.4f}")
    
    # Aplicar el physics model para generar la imagen ruidosa
    with torch.no_grad():
        y_tensor_scaled = physics(x_tensor)
    
    # Convertir de vuelta al rango [0,1] dividiendo por gamma
    y_tensor = y_tensor_scaled / gamma
    y_tensor = torch.clamp(y_tensor, 0.0, 1.0)
    
    # Convertir de vuelta a numpy
    y = y_tensor.squeeze().cpu().numpy()
    
    print(f"Imagen con ruido - min: {y.min():.4f}, max: {y.max():.4f}")
    
    return y
