# Estructura Modular del Proyecto PnP-ADMM

## Descripción
Este proyecto implementa un algoritmo Plug-and-Play ADMM para restauración de imágenes en baja luminosidad usando un modelo de ruido Poisson combinado con un prior de Deep Denoiser.

## Estructura de Archivos

```
src/
├── __init__.py          # Inicialización del paquete
├── config.py            # Configuración y parámetros globales
├── data_loader.py       # Carga y procesamiento de imágenes
├── noise_model.py       # Modelo de ruido Poisson con DeepInverse
├── denoiser.py          # Configuración del modelo Restormer
├── admm_solver.py       # Implementación del algoritmo ADMM
├── metrics.py           # Cálculo de métricas de calidad (PSNR, MSE, MAE)
└── visualization.py     # Funciones de visualización y guardado
```

## Uso

### Ejecución Principal
```bash
python main_modular.py
```

### Usar Módulos Individuales
```python
from src.config import GAMMA
from src.data_loader import load_and_preprocess_image
from src.noise_model import apply_poisson_noise
from src.denoiser import load_denoiser
from src.admm_solver import admm_solver
from src.metrics import print_metrics
from src.visualization import create_results_figure

# Cargar imagen
image = load_and_preprocess_image("mi_imagen.png")

# Aplicar ruido
noisy = apply_poisson_noise(image, GAMMA)

# Cargar denoiser
denoiser = load_denoiser()

# Ejecutar ADMM
restored, residuals = admm_solver(noisy, GAMMA, denoiser)

# Métricas y visualización
print_metrics(image, noisy, restored)
create_results_figure(image, noisy, restored, residuals)
```

## Módulos

### config.py
- **Propósito**: Configuración centralizada de parámetros
- **Parámetros principales**:
  - `GAMMA`: Factor de exposición Poisson (10.0)
  - `RHO`: Parámetro de penalización ADMM (0.1)
  - `MAX_ITER`: Máximo de iteraciones (300)
  - `DEVICE`: Dispositivo de cómputo (CPU/GPU)

### data_loader.py
- **Funciones principales**:
  - `load_and_preprocess_image()`: Carga, convierte a escala de grises, redimensiona y normaliza
  - `validate_image_range()`: Valida rangos de valores

### noise_model.py
- **Funciones principales**:
  - `create_poisson_physics()`: Crea modelo de física de ruido Poisson
  - `apply_poisson_noise()`: Aplica ruido Poisson a una imagen

### denoiser.py
- **Funciones principales**:
  - `load_denoiser()`: Carga modelo Restormer preentrenado
  - `apply_denoiser()`: Aplica denoiser como operador proximal

### admm_solver.py
- **Funciones principales**:
  - `paso_x()`: Actualización x (resuelve ecuación cuadrática)
  - `paso_z()`: Actualización z (aplica denoiser)
  - `paso_u()`: Actualización u (multiplicadores de Lagrange)
  - `admm_solver()`: Algoritmo ADMM completo

### metrics.py
- **Funciones principales**:
  - `calculate_psnr()`: Cálculo de PSNR
  - `calculate_mse()`: Error cuadrático medio
  - `calculate_mae()`: Error absoluto medio
  - `normalize_image()`: Normalización de imagen
  - `print_metrics()`: Reporte completo de métricas

### visualization.py
- **Funciones principales**:
  - `save_single_image()`: Guardar imagen individual
  - `plot_convergence()`: Gráfica de convergencia ADMM
  - `create_results_figure()`: Figura completa con todos los resultados
  - `save_all_individual_images()`: Guardar todas las imágenes

## Ventajas de la Estructura Modular

1. **Mantenibilidad**: Cada módulo tiene responsabilidades claras
2. **Reutilización**: Los módulos se pueden usar independientemente
3. **Escalabilidad**: Fácil agregar nuevas funcionalidades
4. **Testing**: Cada módulo se puede probar por separado
5. **Legibilidad**: Código más organizado y fácil de entender
6. **Configuración centralizada**: Parámetros en un solo lugar

## Archivos Generados

- `imagen_original.png`: Imagen original procesada
- `imagen_ruidosa.png`: Imagen con ruido Poisson aplicado
- `imagen_restaurada.png`: Imagen restaurada por ADMM
- `convergencia_admm.png`: Gráfica de convergencia del algoritmo
- `resultados_completos.png`: Figura completa con análisis

## Dependencias

- numpy
- matplotlib
- opencv-python (cv2)
- torch
- deepinv
- deepinv.models (Restormer)
