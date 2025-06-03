"""
Cálculo de métricas de calidad de imagen
"""

import numpy as np

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcular PSNR entre dos imágenes
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        
    Returns:
        Valor PSNR en dB
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcular Error Cuadrático Medio (MSE)
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        
    Returns:
        Valor MSE
    """
    return np.mean((img1 - img2) ** 2)

def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcular Error Absoluto Medio (MAE)
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        
    Returns:
        Valor MAE
    """
    return np.mean(np.abs(img1 - img2))

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizar imagen al rango [0,1]
    
    Args:
        image: Imagen a normalizar
        
    Returns:
        Imagen normalizada
    """
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val == 0:
        return image
    return (image - min_val) / (max_val - min_val)

def print_metrics(original: np.ndarray, noisy: np.ndarray, restored: np.ndarray):
    """
    Imprimir métricas de calidad comparativas
    
    Args:
        original: Imagen original
        noisy: Imagen con ruido
        restored: Imagen restaurada
    """
    metrics = calculate_metrics(original, noisy, restored)
    
    print("\n" + "="*60)
    print("MÉTRICAS DE CALIDAD:")
    print("="*60)
    print(f"PSNR imagen ruidosa:    {metrics['psnr_noisy']:.2f} dB")
    print(f"PSNR imagen restaurada: {metrics['psnr_restored']:.2f} dB")
    print(f"Mejora en PSNR:         {metrics['psnr_improvement']:.2f} dB")
    print("-"*60)
    print(f"MSE imagen ruidosa:     {metrics['mse_noisy']:.6f}")
    print(f"MSE imagen restaurada:  {metrics['mse_restored']:.6f}")
    print(f"Reducción MSE:          {((metrics['mse_noisy'] - metrics['mse_restored'])/metrics['mse_noisy'])*100:.1f}%")
    print("-"*60)
    print(f"MAE imagen ruidosa:     {metrics['mae_noisy']:.6f}")
    print(f"MAE imagen restaurada:  {metrics['mae_restored']:.6f}")
    print(f"Reducción MAE:          {((metrics['mae_noisy'] - metrics['mae_restored'])/metrics['mae_noisy'])*100:.1f}%")
    print("="*60)
    
    return metrics

def calculate_metrics(original: np.ndarray, noisy: np.ndarray, restored: np.ndarray) -> dict:
    """
    Calcular métricas de calidad comparativas
    
    Args:
        original: Imagen original
        noisy: Imagen con ruido
        restored: Imagen restaurada
        
    Returns:
        Diccionario con las métricas calculadas
    """
    # Normalizar imagen restaurada para métricas justas
    restored_norm = normalize_image(restored)
    
    psnr_noisy = calculate_psnr(original, noisy)
    psnr_restored = calculate_psnr(original, restored_norm)
    
    mse_noisy = calculate_mse(original, noisy)
    mse_restored = calculate_mse(original, restored_norm)
    
    mae_noisy = calculate_mae(original, noisy)
    mae_restored = calculate_mae(original, restored_norm)
    
    return {
        'psnr_noisy': psnr_noisy,
        'psnr_restored': psnr_restored,
        'psnr_improvement': psnr_restored - psnr_noisy,
        'mse_noisy': mse_noisy,
        'mse_restored': mse_restored,
        'mae_noisy': mae_noisy,
        'mae_restored': mae_restored
    }
