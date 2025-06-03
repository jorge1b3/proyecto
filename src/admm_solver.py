"""
Implementación del algoritmo ADMM para Poisson + Deep Denoiser
"""

import torch
import numpy as np
from typing import Tuple, List
from config import DEVICE, RHO, MAX_ITER, TOLERANCE, PRINT_EVERY
from denoiser import apply_denoiser

def paso_x(y: torch.Tensor, gamma: float, rho: float, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Paso x-update: Resolver la ecuación cuadrática pixel a pixel
    
    Resuelve: x = argmin[γ*x - y*log(γ*x) + (ρ/2)||x - z + u||²]
    
    La ecuación cuadrática es: ρx² + (γ - ρz + ρu)x - y = 0
    
    Args:
        y: Imagen ruidosa (tensor) - debe estar escalada por gamma
        gamma: Parámetro del ruido Poisson
        rho: Parámetro de penalización ADMM
        z: Variable auxiliar actual
        u: Variable dual actual
        
    Returns:
        x actualizado
    """
    a = rho
    b = gamma - rho * z + rho * u
    c = -y
    
    # Fórmula cuadrática (tomar raíz positiva)
    discriminant = b**2 - 4*a*c
    x_k = (-b + torch.sqrt(torch.clamp(discriminant, min=1e-10))) / (2*a)
    x_k = torch.clamp(x_k, min=1e-8)  # Evitar valores negativos o cero
    
    return x_k

def paso_z(x: torch.Tensor, u: torch.Tensor, denoiser) -> torch.Tensor:
    """
    Paso z-update: Aplicar el denoiser como operador proximal
    
    Args:
        x: Imagen actual
        u: Variable dual actual
        denoiser: Modelo de deep denoiser
        
    Returns:
        z actualizado
    """
    return apply_denoiser(x, u, denoiser)

def paso_u(x: torch.Tensor, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Paso u-update: Actualizar los multiplicadores de Lagrange
    
    u = u + (x - z)
    
    Args:
        x: Imagen actual
        z: Variable auxiliar actual
        u: Variable dual actual
        
    Returns:
        u actualizado
    """
    return u + (x - z)

def admm_solver(y: np.ndarray, gamma: float, denoiser, 
                rho: float = RHO, max_iter: int = MAX_ITER, 
                tol: float = TOLERANCE, print_every: int = PRINT_EVERY) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Implementación del algoritmo ADMM para Poisson + Deep Denoiser
    
    Args:
        y: Imagen ruidosa
        gamma: Parámetro del ruido Poisson
        denoiser: Modelo denoiser
        rho: Parámetro de penalización ADMM
        max_iter: Máximo número de iteraciones
        tol: Tolerancia para convergencia
        print_every: Frecuencia de impresión de estadísticas
        
    Returns:
        Tuple con (imagen_restaurada, residuales)
    """
    # Escalar y para el modelo Poisson (multiplicar por gamma)
    y_scaled = y * gamma
    y_tensor = torch.from_numpy(y_scaled).to(DEVICE).float()
    
    # Inicialización en el rango [0,1]
    x = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    z = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    u = torch.zeros_like(x)
    
    print(f"Iniciando ADMM con rho={rho}, max_iter={max_iter}")
    print(f"y_scaled range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
    
    residuals = []
    
    for k in range(max_iter):
        x_old = x.clone()
        
        # x-update
        x = paso_x(y_tensor, gamma, rho, z, u)
        
        # z-update
        z = paso_z(x, u, denoiser)
        
        # u-update
        u = paso_u(x, z, u)
        
        # Calcular residuos para convergencia
        residual_primal = torch.norm(x - z).item()
        residual_dual = rho * torch.norm(x - x_old).item()
        residuals.append((residual_primal, residual_dual))
        
        if k % print_every == 0:
            # Calcular el costo con valores escalados correctamente
            x_scaled_for_cost = x * gamma
            cost = torch.sum(x_scaled_for_cost - y_tensor * torch.log(x_scaled_for_cost + 1e-10)).item()
            print(f"Iter {k}: Residual primal = {residual_primal:.6f}, "
                  f"dual = {residual_dual:.6f}, Costo = {cost:.4f}")
        
        # Criterio de convergencia
        if residual_primal < tol and residual_dual < tol:
            print(f"Convergencia alcanzada en iteración {k}")
            break
    
    return x.detach().cpu().numpy(), residuals
