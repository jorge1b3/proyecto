"""
Funciones de visualización y guardado de resultados
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from .config import DPI, FIGSIZE
from .metrics import normalize_image

def save_single_image(image: np.ndarray, title: str, filename: str, 
                     cmap: str = 'gray', dpi: int = DPI) -> None:
    """
    Guardar una imagen individual
    
    Args:
        image: Array de imagen
        title: Título de la imagen
        filename: Nombre del archivo
        cmap: Mapa de colores
        dpi: Resolución
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.title(title)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Imagen guardada: {filename}")

def plot_convergence(residuals: List[Tuple[float, float]], filename: str = 'convergencia.png') -> None:
    """
    Graficar la convergencia del algoritmo ADMM
    
    Args:
        residuals: Lista de tuplas (residual_primal, residual_dual)
        filename: Nombre del archivo
    """
    if not residuals:
        print("No hay datos de residuales para graficar")
        return
    
    primal_res = [r[0] for r in residuals]
    dual_res = [r[1] for r in residuals]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(primal_res, 'b-', label='Residual Primal', linewidth=2)
    plt.semilogy(dual_res, 'r-', label='Residual Dual', linewidth=2)
    plt.xlabel('Iteración')
    plt.ylabel('Residual (escala log)')
    plt.title('Convergencia ADMM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Gráfica de convergencia guardada: {filename}")

def create_results_figure(original: np.ndarray, noisy: np.ndarray, restored: np.ndarray,
                         residuals: Optional[List[Tuple[float, float]]] = None,
                         filename: str = 'resultados_completos.png') -> None:
    """
    Crear figura completa con todos los resultados
    
    Args:
        original: Imagen original
        noisy: Imagen con ruido
        restored: Imagen restaurada
        residuals: Datos de convergencia (opcional)
        filename: Nombre del archivo
    """
    # Normalizar imagen restaurada para visualización
    restored_norm = normalize_image(restored)
    
    # Configurar subplots
    rows = 2 if residuals else 1
    cols = 4 if residuals else 3
    fig, axes = plt.subplots(rows, cols, figsize=FIGSIZE)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Fila superior: imágenes
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(noisy, cmap='gray')
    axes[0,1].set_title('Con Ruido Poisson')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(restored_norm, cmap='gray')
    axes[0,2].set_title('Restaurada (ADMM)')
    axes[0,2].axis('off')
    
    if residuals:
        # Error absoluto
        error_map = np.abs(original - restored_norm)
        im = axes[0,3].imshow(error_map, cmap='hot')
        axes[0,3].set_title('Error Absoluto')
        axes[0,3].axis('off')
        plt.colorbar(im, ax=axes[0,3], fraction=0.046, pad=0.04)
        
        # Fila inferior: análisis
        if len(residuals) > 0:
            primal_res = [r[0] for r in residuals]
            dual_res = [r[1] for r in residuals]
            
            # Convergencia
            axes[1,0].semilogy(primal_res, 'b-', label='Primal', linewidth=2)
            axes[1,0].semilogy(dual_res, 'r-', label='Dual', linewidth=2)
            axes[1,0].set_title('Convergencia ADMM')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_xlabel('Iteración')
            axes[1,0].set_ylabel('Residual')
            
            # Histograma original
            axes[1,1].hist(original.flatten(), bins=50, alpha=0.7, color='blue', density=True)
            axes[1,1].set_title('Histograma Original')
            axes[1,1].set_xlabel('Intensidad')
            axes[1,1].set_ylabel('Densidad')
            
            # Histograma restaurada
            axes[1,2].hist(restored_norm.flatten(), bins=50, alpha=0.7, color='green', density=True)
            axes[1,2].set_title('Histograma Restaurada')
            axes[1,2].set_xlabel('Intensidad')
            axes[1,2].set_ylabel('Densidad')
            
            # Scatter plot: original vs restaurada
            sample_indices = np.random.choice(original.size, 5000, replace=False)
            axes[1,3].scatter(original.flatten()[sample_indices], 
                            restored_norm.flatten()[sample_indices], 
                            alpha=0.5, s=1)
            axes[1,3].plot([0, 1], [0, 1], 'r--', linewidth=2)
            axes[1,3].set_xlabel('Original')
            axes[1,3].set_ylabel('Restaurada')
            axes[1,3].set_title('Original vs Restaurada')
            axes[1,3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Figura de resultados guardada: {filename}")

def save_all_individual_images(original: np.ndarray, noisy: np.ndarray, restored: np.ndarray) -> None:
    """
    Guardar todas las imágenes individuales
    
    Args:
        original: Imagen original
        noisy: Imagen con ruido
        restored: Imagen restaurada
    """
    save_single_image(original, 'Imagen Original', 'imagen_original.png')
    save_single_image(noisy, 'Imagen con Ruido Poisson', 'imagen_ruidosa.png')
    save_single_image(normalize_image(restored), 'Imagen Restaurada', 'imagen_restaurada.png')
