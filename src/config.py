"""
Configuración y parámetros del proyecto PnP-ADMM
"""

import torch

# Parámetros del modelo de ruido Poisson
GAMMA = 10.0  # Factor de exposición/ganancia

# Parámetros ADMM
RHO = 0.1  # Parámetro de penalización
MAX_ITER = 300  # Máximo número de iteraciones
TOLERANCE = 1e-6  # Tolerancia para convergencia

# Configuración del dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuración de imagen
IMAGE_SIZE = 512  # Tamaño de redimensionamiento

# Rutas de archivos
IMAGE_PATH = "lena567.png"
OUTPUT_DIR = "resultados/"

# Configuración de visualización
DPI = 150
FIGSIZE = (15, 10)

# Configuración del denoiser
DENOISER_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "pretrained": "denoising_gray"
}

# Parámetros de convergencia
PRINT_EVERY = 5  # Imprimir estadísticas cada N iteraciones

print(f"Configuración cargada. Dispositivo: {DEVICE}")
