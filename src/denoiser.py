"""
Configuración y carga del modelo denoiser
"""

import torch
from deepinv.models import DRUNet
from config import DEVICE, DENOISER_CONFIG

def load_denoiser() -> DRUNet:
    """
    Cargar y configurar el modelo Restormer para denoising
    
    Returns:
        Modelo Restormer configurado
    """
    print("Cargando modelo Restormer...")
    
    denoiser = DRUNet(
        in_channels=DENOISER_CONFIG["in_channels"],
        out_channels=DENOISER_CONFIG["out_channels"],
        pretrained=DENOISER_CONFIG["pretrained"],
        device=DEVICE,
    )
    
    print("Modelo Restormer cargado exitosamente")
    return denoiser

def apply_denoiser(x: torch.Tensor, u: torch.Tensor, denoiser: DRUNet, sigma: float = 0.03) -> torch.Tensor:
    """
    Aplicar el denoiser como operador proximal
    
    Args:
        x: Imagen actual
        u: Variable dual actual
        denoiser: Modelo denoiser
        sigma: Nivel de ruido del denoiser
        
    Returns:
        z actualizado
    """
    # Combinar x + u
    arg = x + u
    
    # Asegurar que esté en rango válido [0,1] para el denoiser
    arg = torch.clamp(arg, 0.0, 1.0)
    
    # Aplicar denoiser (necesita dimensiones para batch y canal)
    if arg.dim() == 2:
        arg = arg.unsqueeze(0).unsqueeze(0)
    
    sigma_tensor = torch.tensor(sigma)
    
    with torch.no_grad():
        z_k = denoiser(arg, sigma_tensor)
    
    # Remover dimensiones extra y retornar
    z_k = z_k.squeeze()
    z_k = torch.clamp(z_k, 0.0, 1.0)  # Mantener en [0,1]
    
    return z_k
