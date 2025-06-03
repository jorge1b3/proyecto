# Solución Matemática: PnP-ADMM para Restauración de Imágenes en Baja Luminosidad

## Autores
- **Miguel Fernando Pimiento Escobar**
- **Jorge Andrey Gracia Vega**

---

## 1. Planteamiento del Problema

### 1.1 Modelo de Observación

En condiciones de **baja luminosidad**, el modelo de adquisición de imágenes sigue una distribución de **Poisson**:

$$y_i \sim \text{Poisson}(\gamma \cdot x_i), \quad i = 1, 2, \ldots, m$$

donde:
- $y \in \mathbb{N}^m$ es la imagen observada (ruidosa) vectorizada
- $x \in \mathbb{R}_+^m$ es la imagen original que queremos recuperar
- $\gamma > 0$ es el **factor de exposición** o ganancia del sensor
- $m$ es el número total de píxeles

### 1.2 Formulación del Problema de Optimización

El objetivo es resolver el problema de optimización:

$$\min_{x \geq 0} \left\{ f(x) + g(x) \right\}$$

donde:

- **Término de fidelidad a los datos** (neg-log-verosimilitud Poisson):
  $$f(x) = \gamma \mathbf{1}^T x - y^T \log(\gamma x)$$

- **Término de regularización** (prior implícito):
  $$g(x) = \lambda R(x)$$

donde $R(x)$ es un **regularizador implícito** definido a través de un **Deep Denoiser** preentrenado $\mathcal{D}(\cdot)$.

---

## 2. Reformulación con Variable Auxiliar

Para aplicar **ADMM**, introducimos una **variable auxiliar** $z$ y la restricción de igualdad $x = z$:

$$\begin{align}
\min_{x,z} \quad & f(x) + g(z) \\
\text{sujeto a} \quad & x = z
\end{align}$$

Esta reformulación permite separar los términos difíciles de optimizar:
- $f(x)$: término no-suave pero con estructura conocida
- $g(z)$: término implícito que se resuelve vía denoiser

---

## 3. Lagrangiano Aumentado

El **Lagrangiano aumentado escalado** con parámetro de penalización $\rho > 0$ es:

$$\mathcal{L}_\rho(x, z, u) = f(x) + g(z) + \frac{\rho}{2} \|x - z + u\|_2^2$$

donde $u$ es el **multiplicador de Lagrange escalado** (variable dual).

---

## 4. Algoritmo ADMM

Las iteraciones de **ADMM** alternan entre tres pasos:

### 4.1 Paso x-update

$$x^{k+1} = \arg\min_{x \geq 0} \left\{ f(x) + \frac{\rho}{2} \|x - z^k + u^k\|_2^2 \right\}$$

### 4.2 Paso z-update (Plug-and-Play)

$$z^{k+1} = \arg\min_z \left\{ g(z) + \frac{\rho}{2} \|x^{k+1} - z + u^k\|_2^2 \right\} = \mathcal{D}(x^{k+1} + u^k)$$

### 4.3 Paso u-update

$$u^{k+1} = u^k + (x^{k+1} - z^{k+1})$$

---

## 5. Derivación del Paso x-update

### 5.1 Problema de Optimización

El paso x-update requiere resolver:

$$x^{k+1} = \arg\min_{x \geq 0} \left\{ \gamma \mathbf{1}^T x - y^T \log(\gamma x) + \frac{\rho}{2} \|x - z^k + u^k\|_2^2 \right\}$$

### 5.2 Condición de Optimalidad

Diferenciando el objetivo respecto a $x_i$ (componente $i$-ésima):

$$\frac{\partial}{\partial x_i} \left[ \gamma x_i - y_i \log(\gamma x_i) + \frac{\rho}{2}(x_i - z_i^k + u_i^k)^2 \right] = 0$$

### 5.3 Cálculo de la Derivada

$$\frac{\partial}{\partial x_i} = \gamma - y_i \frac{1}{\gamma x_i} \cdot \gamma + \rho(x_i - z_i^k + u_i^k)$$

$$= \gamma - \frac{y_i}{x_i} + \rho(x_i - z_i^k + u_i^k)$$

### 5.4 Ecuación de Optimalidad

Igualando a cero:

$$\gamma - \frac{y_i}{x_i} + \rho(x_i - z_i^k + u_i^k) = 0$$

Multiplicando por $x_i$:

$$\gamma x_i - y_i + \rho x_i(x_i - z_i^k + u_i^k) = 0$$

$$\gamma x_i - y_i + \rho x_i^2 - \rho z_i^k x_i + \rho u_i^k x_i = 0$$

### 5.5 Ecuación Cuadrática

Reordenando:

$$\rho x_i^2 + (\gamma - \rho z_i^k + \rho u_i^k) x_i - y_i = 0$$

Esta es una **ecuación cuadrática** de la forma $ax^2 + bx + c = 0$ con:
- $a = \rho$
- $b = \gamma - \rho z_i^k + \rho u_i^k$
- $c = -y_i$

### 5.6 Solución Analítica

Usando la fórmula cuadrática y tomando la **raíz positiva** (ya que $x_i \geq 0$):

$$x_i^{k+1} = \frac{-b + \sqrt{b^2 - 4ac}}{2a}$$

$$= \frac{-(\gamma - \rho z_i^k + \rho u_i^k) + \sqrt{(\gamma - \rho z_i^k + \rho u_i^k)^2 + 4\rho y_i}}{2\rho}$$

---

## 6. Implementación del Paso z-update

### 6.1 Operador Proximal

El paso z-update se resuelve como:

$$z^{k+1} = \text{prox}_{g/\rho}(x^{k+1} + u^k)$$

### 6.2 Plug-and-Play

En el esquema **Plug-and-Play**, reemplazamos el operador proximal exacto por un **denoiser preentrenado**:

$$z^{k+1} = \mathcal{D}(x^{k+1} + u^k)$$

donde $\mathcal{D}$ es un modelo de **deep learning** (e.g., Restormer) entrenado para denoising.

### 6.3 Justificación Teórica

Esta aproximación es válida bajo ciertas condiciones de regularidad del denoiser, y funciona bien en la práctica cuando:
1. El denoiser está bien entrenado
2. El parámetro $\rho$ está adecuadamente ajustado
3. La inicialización es apropiada

---

## 7. Análisis de Convergencia

### 7.1 Residuos ADMM

Para monitorear la convergencia, calculamos:

- **Residuo primal**: $r^k = \|x^k - z^k\|_2$
- **Residuo dual**: $s^k = \rho \|z^k - z^{k-1}\|_2$

### 7.2 Criterio de Parada

El algoritmo converge cuando:

$$r^k < \epsilon_{\text{pri}} \quad \text{y} \quad s^k < \epsilon_{\text{dual}}$$

donde $\epsilon_{\text{pri}}$ y $\epsilon_{\text{dual}}$ son tolerancias predefinidas.

---

## 8. Parámetros del Algoritmo

### 8.1 Parámetros Principales

- **$\gamma$**: Factor de exposición (típicamente 10.0)
- **$\rho$**: Parámetro de penalización ADMM (típicamente 0.1)
- **$\epsilon$**: Tolerancia de convergencia ($10^{-6}$)
- **Máx. iter**: Número máximo de iteraciones (300)

### 8.2 Ajuste de Parámetros

- **$\rho$ pequeño**: Convergencia lenta pero estable
- **$\rho$ grande**: Convergencia rápida pero posible inestabilidad
- **$\gamma$**: Determinado por las condiciones de adquisición

---

## 9. Propiedades Matemáticas

### 9.1 Convexidad

- $f(x)$ es **convexa** en $x \geq 0$ (neg-log-verosimilitud Poisson)
- El término cuadrático $\frac{\rho}{2}\|x - z + u\|_2^2$ es **estrictamente convexo**
- El paso x-update tiene **solución única**

### 9.2 Diferenciabilidad

- $f(x)$ es **diferenciable** en $x > 0$
- La derivada tiene **forma cerrada**
- Permite solución **analítica** del paso x-update

### 9.3 Escalado

- El problema mantiene su estructura bajo **escalado lineal**
- La normalización es crucial para la **estabilidad numérica**
- Los rangos de valores deben ser **cuidadosamente manejados**

---

## 10. Implementación Numérica

### 10.1 Consideraciones de Estabilidad

1. **Clipping**: $x_i = \max(x_i, \epsilon)$ para evitar $\log(0)$
2. **Discriminante**: $\Delta = \max(\Delta, \epsilon)$ en la fórmula cuadrática
3. **Normalización**: Mantener imágenes en rango $[0,1]$

### 10.2 Escalado del Modelo Poisson

```math
\begin{align}
\text{Entrada al modelo:} \quad &\tilde{x} = \gamma \cdot x \\
\text{Salida del modelo:} \quad &\tilde{y} = \text{Poisson}(\tilde{x}) \\
\text{Normalización:} \quad &y = \tilde{y} / \gamma
\end{align}
```

### 10.3 Manejo del Denoiser

- **Entrada**: Imagen en rango $[0,1]$
- **Salida**: Imagen restaurada en rango $[0,1]$
- **Dimensiones**: Conversión adecuada para modelos CNN

---

## 11. Métricas de Evaluación

### 11.1 PSNR (Peak Signal-to-Noise Ratio)

$$\text{PSNR} = 20 \log_{10} \left( \frac{\text{MAX}}{\sqrt{\text{MSE}}} \right)$$

donde $\text{MAX} = 1$ para imágenes normalizadas.

### 11.2 MSE (Mean Squared Error)

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^m (x_i - \hat{x}_i)^2$$

### 11.3 MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^m |x_i - \hat{x}_i|$$

---

## Referencias

1. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). *Distributed optimization and statistical learning via the alternating direction method of multipliers*. Foundations and Trends in Machine learning, 3(1), 1-122.

2. Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). *Plug-and-play priors for model based reconstruction*. IEEE Global Conference on Signal and Information Processing.

3. Zhang, K., Zuo, W., Gu, S., & Zhang, L. (2017). *Learning deep CNN denoiser prior for image restoration*. IEEE Conference on Computer Vision and Pattern Recognition.

4. Ahmad, R., Bouman, C. A., Buzzard, G. T., Chan, S., Liu, S., Reehorst, E. T., & Schniter, P. (2020). *Plug-and-play methods for magnetic resonance imaging: Using denoisers for image recovery*. IEEE Signal Processing Magazine, 37(1), 105-116.
