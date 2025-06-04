"""
Configuración y parámetros del proyecto PnP-ADMM
"""

import torch
import argparse
import os

# Configuración por defecto
GAMMA = 10.0
RHO = 0.1
MAX_ITER = 300
TOLERANCE = 1e-6
IMAGE_SIZE = 512
IMAGE_PATH = "lena567.png"
OUTPUT_DIR = "resultados/"
PRINT_EVERY = 5

# Configuración de visualización
DPI = 150
FIGSIZE = (15, 10)


# Configuración del denoiser
DENOISER_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "pretrained": "download"
}

# Parámetro sigma para el denoiser
SIGMA_DENOISER = 0.03  # Nivel de ruido para el denoiser

# Configuración del dispositivo
def get_device(device_arg="auto"):
    """
    Determina el dispositivo de cómputo a usar
    
    Args:
        device_arg (str): 'auto', 'cpu', o 'cuda'
        
    Returns:
        str: dispositivo a usar
    """
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg

DEVICE = get_device()

def parse_arguments():
    """
    Parser de argumentos de línea de comandos
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="PnP-ADMM para restauración de imágenes en baja luminosidad",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parámetros del algoritmo
    parser.add_argument(
        "--gamma", type=float, default=GAMMA,
        help="Factor de exposición/ganancia del modelo Poisson"
    )
    parser.add_argument(
        "--rho", type=float, default=RHO,
        help="Parámetro de penalización ADMM"
    )
    parser.add_argument(
        "--max-iter", type=int, default=MAX_ITER,
        help="Número máximo de iteraciones ADMM"
    )
    parser.add_argument(
        "--tolerance", type=float, default=TOLERANCE,
        help="Tolerancia para convergencia"
    )
    parser.add_argument(
        "--sigma", type=float, default=SIGMA_DENOISER,
        help="Nivel de ruido del denoiser (sigma)"
    )
    
    # Configuración de imagen
    parser.add_argument(
        "--image", type=str, default=IMAGE_PATH,
        help="Ruta de la imagen de entrada"
    )
    parser.add_argument(
        "--size", type=int, default=IMAGE_SIZE,
        help="Tamaño de redimensionamiento de la imagen"
    )
    
    # Configuración de salida
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Directorio de salida para resultados"
    )
    parser.add_argument(
        "--print-every", type=int, default=PRINT_EVERY,
        help="Imprimir estadísticas cada N iteraciones"
    )
    
    # Configuración del dispositivo
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo de cómputo (auto detecta GPU si está disponible)"
    )
    
    # Configuración de visualización
    parser.add_argument(
        "--dpi", type=int, default=DPI,
        help="DPI para las imágenes guardadas"
    )
    parser.add_argument(
        "--no-save-individual", action="store_true",
        help="No guardar imágenes individuales"
    )
    parser.add_argument(
        "--no-save-plots", action="store_true",
        help="No guardar gráficas de convergencia"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Modo verboso con información detallada"
    )
    
    # Configuración del denoiser
    parser.add_argument(
        "--denoiser-model", type=str, 
        default=DENOISER_CONFIG['pretrained'],
        choices=['download'],
        help="Modelo del denoiser Restormer"
    )
    
    return parser.parse_args()

def update_config_from_args(args):
    """
    Actualiza las variables globales de configuración con los argumentos parseados
    
    Args:
        args (argparse.Namespace): Argumentos parseados
    """
    global GAMMA, RHO, MAX_ITER, TOLERANCE, DEVICE, IMAGE_SIZE
    global IMAGE_PATH, OUTPUT_DIR, DPI, DENOISER_CONFIG, PRINT_EVERY
    
    # Actualizar parámetros del algoritmo
    GAMMA = args.gamma
    RHO = args.rho
    MAX_ITER = args.max_iter
    TOLERANCE = args.tolerance
    
    # Actualizar configuración de imagen
    IMAGE_PATH = args.image
    IMAGE_SIZE = args.size
    
    # Actualizar configuración de salida
    OUTPUT_DIR = args.output_dir
    DPI = args.dpi
    PRINT_EVERY = args.print_every
    
    # Configurar dispositivo
    DEVICE = get_device(args.device)
    
    # Actualizar configuración del denoiser
    DENOISER_CONFIG['pretrained'] = args.denoiser_model
    
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Mostrar configuración si el modo verboso está activado
    if args.verbose:
        print_config_summary(args)

def print_config_summary(args):
    """
    Imprime un resumen de la configuración actual
    
    Args:
        args (argparse.Namespace): Argumentos parseados
    """
    print("\n" + "="*60)
    print("CONFIGURACIÓN DEL ALGORITMO PnP-ADMM")
    print("="*60)
    print("Parámetros del algoritmo:")
    print(f"  Gamma (exposición):     {GAMMA}")
    print(f"  Rho (penalización):     {RHO}")
    print(f"  Máx. iteraciones:       {MAX_ITER}")
    print(f"  Tolerancia:             {TOLERANCE}")
    print(f"  Sigma denoiser:         {args.sigma}")
    print("\nConfiguración de imagen:")
    print(f"  Imagen de entrada:      {IMAGE_PATH}")
    print(f"  Tamaño de imagen:       {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("\nConfiguración de salida:")
    print(f"  Directorio de salida:   {OUTPUT_DIR}")
    print(f"  DPI de figuras:         {DPI}")
    print("\nConfiguración de ejecución:")
    print(f"  Dispositivo:            {DEVICE}")
    print(f"  Modelo denoiser:        {DENOISER_CONFIG['pretrained']}")
    print(f"  Modo verboso:           {args.verbose}")
    print("="*60)

# Información al cargar el módulo
if __name__ != "__main__":
    print(f"Configuración cargada. Dispositivo por defecto: {DEVICE}")