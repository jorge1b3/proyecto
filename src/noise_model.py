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
        gamma: Parámetro de ganancia (nivel de ruido inverso - más pequeño = más ruido)
        
    Returns:
        Imagen con ruido Poisson en rango [0,1]
    """
    # Para gamma muy pequeño, usamos un enfoque directo del ruido Poisson
    # sin usar DeepInverse para evitar problemas de escalado
    
    if gamma < 0.01:  # Para valores muy pequeños de gamma
        # Implementación directa del ruido Poisson
        # gamma controla la intensidad: más pequeño = más ruido
        
        # Escalar para tener valores razonables para Poisson
        scale_factor = 100.0  # Factor de escalado para evitar valores muy pequeños
        x_scaled = image * scale_factor
        
        # Aplicar ruido Poisson usando numpy
        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
            
            # Ruido Poisson: cada pixel sigue Poisson(lambda = gamma * pixel_value / gamma) = Poisson(pixel_value)
            # Pero queremos controlar la intensidad con gamma
            lambda_param = x_tensor / gamma  # Más pequeño gamma = más ruido
            
            # Generar muestras Poisson
            y_noisy = torch.poisson(lambda_param)
            
            # Escalar de vuelta, considerando que aplicamos ruido con factor 1/gamma
            y_scaled_back = y_noisy * gamma / scale_factor
            
        # Normalizar suavemente para evitar clipping brusco
        y_clipped = torch.clamp(y_scaled_back, 0.0, 1.0)
        
        print(f"Imagen escalada (factor {scale_factor}) - min: {x_scaled.min():.4f}, max: {x_scaled.max():.4f}")
        print(f"Lambda param - min: {lambda_param.min():.4f}, max: {lambda_param.max():.4f}")
        print(f"Imagen con ruido antes clamp - min: {y_scaled_back.min():.4f}, max: {y_scaled_back.max():.4f}")
        
    else:
        # Para gamma >= 0.01, usar DeepInverse normalmente
        physics = create_poisson_physics(gamma)
        
        x_scaled = image * gamma
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        print(f"Imagen escalada para Poisson - min: {x_tensor.min():.4f}, max: {x_tensor.max():.4f}")
        
        with torch.no_grad():
            y_tensor_scaled = physics(x_tensor)
        
        y_tensor = y_tensor_scaled / gamma
        y_clipped = torch.clamp(y_tensor, 0.0, 1.0).squeeze()
    
    # Convertir a numpy
    y = y_clipped.cpu().numpy()
    
    print(f"Imagen con ruido final - min: {y.min():.4f}, max: {y.max():.4f}")
    print(f"Porcentaje de pixeles en [0, 0.1]: {np.mean(y <= 0.1)*100:.1f}%")
    print(f"Porcentaje de pixeles en [0.9, 1.0]: {np.mean(y >= 0.9)*100:.1f}%")
    
    return y
