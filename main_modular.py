"""
Script principal del proyecto PnP-ADMM para restauración de imágenes en baja luminosidad

Autores: Miguel Fernando Pimiento Escobar, Jorge Andrey Gracia Vega
Curso: Optimización Convexa
"""

import sys
import os
from pathlib import Path

# Agregar src al path para imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config import GAMMA, IMAGE_PATH
from src.data_loader import load_and_preprocess_image, validate_image_range
from src.noise_model import apply_poisson_noise
from src.denoiser import load_denoiser
from src.admm_solver import admm_solver
from src.metrics import print_metrics
from src.visualization import (save_all_individual_images, 
                              create_results_figure, 
                              plot_convergence)

def main():
    """
    Función principal que ejecuta todo el pipeline de restauración
    """
    print("="*70)
    print("PROYECTO PnP-ADMM PARA RESTAURACIÓN DE IMÁGENES EN BAJA LUMINOSIDAD")
    print("Autores: Miguel Fernando Pimiento Escobar, Jorge Andrey Gracia Vega")
    print("="*70)
    
    # 1. Cargar y preprocesar imagen
    print("\n1. CARGANDO IMAGEN ORIGINAL...")
    try:
        original_image = load_and_preprocess_image(IMAGE_PATH)
        validate_image_range(original_image, "Imagen original")
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return
    
    # 2. Aplicar ruido Poisson
    print(f"\n2. APLICANDO RUIDO POISSON (gamma={GAMMA})...")
    noisy_image = apply_poisson_noise(original_image, GAMMA)
    validate_image_range(noisy_image, "Imagen ruidosa")
    
    # 3. Cargar modelo denoiser
    print("\n3. CARGANDO MODELO DENOISER...")
    denoiser = load_denoiser()
    
    # 4. Ejecutar algoritmo ADMM
    print("\n4. EJECUTANDO ALGORITMO ADMM...")
    restored_image, residuals = admm_solver(noisy_image, GAMMA, denoiser)
    validate_image_range(restored_image, "Imagen restaurada")
    
    # 5. Calcular métricas
    print("\n5. CALCULANDO MÉTRICAS DE CALIDAD...")
    metrics = print_metrics(original_image, noisy_image, restored_image)
    
    # 6. Guardar resultados
    print("\n6. GUARDANDO RESULTADOS...")
    
    # Guardar imágenes individuales
    save_all_individual_images(original_image, noisy_image, restored_image)
    
    # Guardar gráfica de convergencia
    plot_convergence(residuals, 'convergencia_admm.png')
    
    # Guardar figura completa de resultados
    create_results_figure(original_image, noisy_image, restored_image, 
                         residuals, 'resultados_completos.png')
    
    # 7. Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS:")
    print("="*70)
    print(f"Imagen original:  {original_image.shape}")
    print(f"Parámetro gamma:  {GAMMA}")
    print(f"Iteraciones ADMM: {len(residuals)}")
    if metrics:
        print(f"Mejora PSNR:      {metrics['psnr_improvement']:.2f} dB")
        print(f"PSNR final:       {metrics['psnr_restored']:.2f} dB")
    print("="*70)
    print("¡Ejecución completada exitosamente!")
    print("Revisa los archivos generados para ver los resultados.")

if __name__ == "__main__":
    main()
