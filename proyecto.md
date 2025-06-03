**Actividad Académica:**
Proyecto final de la materia Optimización Convexa.

**Presentación:**
Esta actividad de final de semestre consiste en implementar un algoritmo ADMM.

**Organización de grupos:**
Cada estudiante escoge un problema de los planteados a continuación, lo mezcla con su prior favorito, y desarrolla el algoritmo ADMM correspondiente. Para esto, cada grupo está conformado por máximo 3 estudiantes para realizar el trabajo colaborativo. Realizar el proyecto de forma individual es también aceptado.


**Low-Light**

* Imagen:
  *(Imagen de Lenna con ruido)*
* $y \sim P(\gamma x)$
* $P$ es ruido Poisson con parámetro $\gamma$
* Modelo:

$$
\gamma 1^T x - y^T \log(\gamma x)
$$

---

**Deep Denoiser**

* $R(x)$
* No se conoce quién es $R$
* $\operatorname{Prox} = D(x)$
* Donde $D$ representa una red de denoiser

---

## Formulación Matemática ADMM

### Problema: Low-Light con Prior Deep Denoiser

**Problema original:**
Combinar el modelo de ruido Poisson (Low-Light) con un prior regularizador basado en Deep Denoiser:

$$
\min_x \quad \gamma 1^T x - y^T \log(\gamma x) + R(x)
$$

donde:
- **Término de fidelidad (Low-Light):** $\gamma 1^T x - y^T \log(\gamma x)$
  - $x$ es la imagen restaurada
  - $y$ es la imagen observada con ruido Poisson: $y \sim P(\gamma x)$
  - $\gamma$ es el parámetro del ruido Poisson
- **Prior regularizador (Deep Denoiser):** $R(x)$
  - $R(x)$ es un regularizador desconocido
  - Solo tenemos acceso al operador proximal: $\text{Prox}_R(x) = D(x)$
  - $D$ representa una red denoiser pre-entrenada

**Formulación ADMM:**
Para aplicar ADMM, reformulamos el problema introduciendo una variable auxiliar $z$:

$$
\begin{align}
\min_{x,z} \quad & \gamma 1^T x - y^T \log(\gamma x) + R(z) \\
\text{s.t.} \quad & x = z
\end{align}
$$

**Lagrangiano aumentado (formulación escalada):**
$$
L_\rho(x,z,u) = \gamma 1^T x - y^T \log(\gamma x) + R(z) + \frac{\rho}{2}\|x-z+u\|_2^2
$$

donde $u = \lambda/\rho$ es la variable dual escalada.

**Algoritmo ADMM:**
1. **x-update (resuelve el modelo de ruido Poisson):** 
   $$x^{k+1} = \arg\min_x \left[ \gamma 1^T x - y^T \log(\gamma x) + \frac{\rho}{2}\|x-z^k+u^k\|_2^2 \right]$$

2. **z-update (usando denoiser como operador proximal):** 
   $$z^{k+1} = D\left(x^{k+1} + u^k\right)$$

3. **u-update (multiplicadores de Lagrange escalados):** 
   $$u^{k+1} = u^k + (x^{k+1} - z^{k+1})$$

**Derivación del x-update:**
Para cada pixel $i$, necesitamos resolver:
$$\min_{x_i} \left[ \gamma x_i - y_i \log(\gamma x_i) + \frac{\rho}{2}(x_i - z_i^k + u_i^k)^2 \right]$$

Derivando e igualando a cero:
$$\gamma - \frac{y_i}{x_i} + \rho(x_i - z_i^k + u_i^k) = 0$$

Multiplicando por $x_i$ y reordenando:
$$\rho x_i^2 + (\gamma - \rho z_i^k + \rho u_i^k)x_i - y_i = 0$$

La solución de esta ecuación cuadrática es:
$$x_i^{k+1} = \frac{-(\gamma - \rho z_i^k + \rho u_i^k) + \sqrt{(\gamma - \rho z_i^k + \rho u_i^k)^2 + 4\rho y_i}}{2\rho}$$