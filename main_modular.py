"""
Script principal del proyecto PnP-ADMM para restauración de imágenes en baja luminosidad

Uso:
    python main_modular.py --image imagen.jpg --gamma 10.0 --rho 0.1
    python main_modular.py --image test.png --max-iter 500 --device cuda
    python main_modular.py --help  # Para ver todas las opciones

Autores: Miguel Fernando Pimiento Escobar, Jorge Andrey Gracia Vega
Curso: Optimización Convexa
"""

import sys
import os
from pathlib import Path

# Agregar src al path para imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importar módulos del proyecto
from config import parse_arguments, update_config_from_args, get_device
from data_loader import load_and_preprocess_image
from noise_model import apply_poisson_noise
from denoiser import load_denoiser
from admm_solver import admm_solver
from metrics import calculate_metrics, print_metrics
from visualization import create_results_figure, plot_convergence, save_all_individual_images

def validate_image_range(image, name):
    """Validar que la imagen esté en rango [0,1]"""
    print(f"   {name} - min: {image.min():.4f}, max: {image.max():.4f}")
    if image.min() < 0 or image.max() > 1:
        print(f"   ⚠️ ADVERTENCIA: {name} fuera del rango [0,1]")

def main():
    """
    Función principal que ejecuta todo el pipeline de restauración
    """
    try:
        # Parsear argumentos de línea de comandos
        args = parse_arguments()
        
        # Actualizar configuración global
        update_config_from_args(args)

        print("Configuración:")
        print(f"  Imagen: {args.image}")
        print(f"  Gamma: {args.gamma}")
        print(f"  Rho: {args.rho}")
        print(f"  Max iteraciones: {args.max_iter}")
        print(f"  Tolerancia: {args.tolerance}")
        print(f"  Dispositivo: {args.device}")
        print(f"  Directorio salida: {args.output_dir}")
        print("="*70)
        
        # Crear directorio de salida si no existe
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 1. Cargar y preprocesar imagen
        print("\\n1. CARGANDO IMAGEN ORIGINAL...")
        try:
            original_image = load_and_preprocess_image(args.image, size=args.size)
            validate_image_range(original_image, "Imagen original")
        except Exception as e:
            print(f"❌ Error cargando imagen: {e}")
            return 1
        
        # 2. Aplicar ruido Poisson
        print(f"\\n2. APLICANDO RUIDO POISSON (gamma={args.gamma})...")
        try:
            noisy_image = apply_poisson_noise(original_image, args.gamma)
            validate_image_range(noisy_image, "Imagen ruidosa")
        except Exception as e:
            print(f"❌ Error aplicando ruido: {e}")
            return 1
        
        # 3. Cargar modelo denoiser
        print("\\n3. CARGANDO MODELO DENOISER...")
        try:
            device = get_device(args.device)
            denoiser = load_denoiser()
            if args.verbose:
                print(f"   Modelo: {args.denoiser_model}")
                print(f"   Dispositivo: {device}")
        except Exception as e:
            print(f"❌ Error cargando denoiser: {e}")
            return 1
        
        # 4. Ejecutar algoritmo ADMM
        print("\\n4. EJECUTANDO ALGORITMO ADMM...")
        try:
            restored_image, residuals = admm_solver(
                noisy_image, 
                args.gamma, 
                denoiser, 
                rho=args.rho,
                max_iter=args.max_iter,
                tol=args.tolerance,
                print_every=args.print_every
            )
            validate_image_range(restored_image, "Imagen restaurada")
        except Exception as e:
            print(f"❌ Error en ADMM: {e}")
            return 1
        
        # 5. Calcular métricas
        print("\\n5. CALCULANDO MÉTRICAS DE CALIDAD...")
        try:
            metrics = calculate_metrics(original_image, noisy_image, restored_image)
            print_metrics(metrics)
        except Exception as e:
            print(f"❌ Error calculando métricas: {e}")
            return 1
        
        # 6. Guardar resultados
        print("\\n6. GUARDANDO RESULTADOS...")
        try:
            # Guardar visualización completa
            create_results_figure(
                original_image, 
                noisy_image, 
                restored_image,
                residuals,
                filename=f"{args.output_dir}/resultados_completos.png"
            )
            
            # Guardar imágenes individuales (si no está deshabilitado)
            if not args.no_save_individual:
                # Cambiar al directorio de salida temporalmente para las imágenes individuales
                original_dir = os.getcwd()
                os.chdir(args.output_dir)
                save_all_individual_images(
                    original_image, 
                    noisy_image, 
                    restored_image
                )
                os.chdir(original_dir)
            
            # Guardar gráfica de convergencia (si no está deshabilitado)
            if not args.no_save_plots:
                plot_convergence(
                    residuals,
                    filename=f"{args.output_dir}/convergencia.png"
                )
                
            print(f"   ✅ Resultados guardados en: {args.output_dir}")
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return 1
        
        # Resumen final
        print("\\n" + "="*70)
        print("¡EJECUCIÓN COMPLETADA EXITOSAMENTE!")
        print(f"Mejora en PSNR: {metrics['psnr_improvement']:.2f} dB")
        print("Revisa los archivos generados para ver los resultados.")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n❌ Ejecución interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\\n❌ Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
