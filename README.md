# Proyecto PnP-ADMM para Restauración de Imágenes en Baja Luminosidad

## Autores
- **Miguel Fernando Pimiento Escobar**
- **Jorge Andrey Gracia Vanegas**

## Descripción
Implementación de un algoritmo **Plug-and-Play ADMM** para restauración de imágenes en condiciones de baja luminosidad, combinando un modelo de ruido **Poisson** con un **Deep Denoiser** como prior implícito.

## Documentación Matemática

### 📚 Documentos Principales

1. **[SOLUCION_MATEMATICA.md](SOLUCION_MATEMATICA.md)** - Formulación completa del problema
   - Planteamiento matemático del problema
   - Derivación detallada del algoritmo ADMM
   - Análisis de convergencia y propiedades teóricas
   - Métricas de evaluación

2. **[IMPLEMENTACION_TECNICA.md](IMPLEMENTACION_TECNICA.md)** - Detalles de implementación
   - Derivación paso a paso del x-update
   - Consideraciones de estabilidad numérica
   - Optimización de rendimiento
   - Debugging y diagnóstico

3. **[ESTRUCTURA_MODULAR.md](ESTRUCTURA_MODULAR.md)** - Organización del código
   - Estructura de módulos
   - Uso de la API
   - Ventajas de la modularización

## Estructura del Proyecto

```
proyecto/
├── README.md                    # Este archivo
├── SOLUCION_MATEMATICA.md      # Formulación matemática completa
├── IMPLEMENTACION_TECNICA.md   # Detalles técnicos de implementación
├── ESTRUCTURA_MODULAR.md       # Documentación de la estructura modular
├── main_modular.py             # Script principal modular
├── test.py                     # Implementación original (monolítica)
├── src/                        # Código fuente modular
│   ├── __init__.py
│   ├── config.py              # Configuración y parámetros
│   ├── data_loader.py         # Carga y procesamiento de imágenes
│   ├── noise_model.py         # Modelo de ruido Poisson
│   ├── denoiser.py            # Configuración del denoiser
│   ├── admm_solver.py         # Algoritmo ADMM
│   ├── metrics.py             # Métricas de calidad
│   └── visualization.py       # Visualización y guardado
└── lena567.png                # Imagen de prueba

Resultados generados:
├── imagen_original.png
├── imagen_ruidosa.png
├── imagen_restaurada.png
├── convergencia_admm.png
└── resultados_completos.png
```

## Uso Rápido

### Ejecución Principal
```bash
python main_modular.py
```

### Uso de Módulos Individuales
```python
from src.config import GAMMA, RHO, MAX_ITER
from src.data_loader import load_and_preprocess_image
from src.noise_model import apply_poisson_noise
from src.denoiser import load_denoiser
from src.admm_solver import admm_solver
from src.metrics import print_metrics
from src.visualization import create_results_figure

# Cargar y procesar imagen
image = load_and_preprocess_image("mi_imagen.png")

# Aplicar ruido Poisson
noisy = apply_poisson_noise(image, GAMMA)

# Cargar denoiser preentrenado
denoiser = load_denoiser()

# Ejecutar algoritmo ADMM
restored, residuals = admm_solver(noisy, GAMMA, denoiser, RHO, MAX_ITER)

# Evaluar y visualizar
print_metrics(image, noisy, restored)
create_results_figure(image, noisy, restored, residuals)
```

## Algoritmo ADMM

El algoritmo resuelve el problema de optimización:

$$\min_{x \geq 0} \left\{ \gamma \mathbf{1}^T x - y^T \log(\gamma x) + \lambda R(x) \right\}$$

mediante las iteraciones:

1. **x-update**: $x^{k+1} = \arg\min_x \left\{ f(x) + \frac{\rho}{2} \|x - z^k + u^k\|_2^2 \right\}$
2. **z-update**: $z^{k+1} = \mathcal{D}(x^{k+1} + u^k)$ (Plug-and-Play)
3. **u-update**: $u^{k+1} = u^k + (x^{k+1} - z^{k+1})$

### Parámetros Principales

- **γ (gamma)**: Factor de exposición Poisson (10.0)
- **ρ (rho)**: Parámetro de penalización ADMM (0.1)
- **Tolerancia**: Criterio de convergencia (1e-6)
- **Max iter**: Número máximo de iteraciones (300)

## Resultados

El algoritmo produce:

- **Restauración efectiva** de imágenes con ruido Poisson
- **Convergencia estable** en ~30-50 iteraciones
- **Mejoras significativas** en PSNR (típicamente +5-15 dB)
- **Preservación de detalles** gracias al deep denoiser

## Dependencias

```bash
pip install numpy matplotlib opencv-python torch deepinv
```

### Versiones Recomendadas
- Python ≥ 3.8
- PyTorch ≥ 1.10
- DeepInverse ≥ 0.10
- CUDA (opcional, para GPU)

## Métricas de Evaluación

- **PSNR**: Peak Signal-to-Noise Ratio
- **MSE**: Mean Squared Error  
- **MAE**: Mean Absolute Error
- **Tiempo de convergencia**: Número de iteraciones hasta convergencia

## Archivos Importantes

### Implementaciones
- `main_modular.py`: Implementación modular recomendada
- `test.py`: Implementación original (monolítica, para referencia)

### Resultados
- `imagen_original.png`: Imagen de entrada procesada
- `imagen_ruidosa.png`: Imagen con ruido Poisson aplicado
- `imagen_restaurada.png`: Resultado del algoritmo ADMM
- `convergencia_admm.png`: Gráficas de convergencia
- `resultados_completos.png`: Panel completo de análisis