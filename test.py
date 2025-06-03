# %% [markdown]
# # Miguel Fernando Pimiento Escobar
# # Jorge Andrey Gracia Vega

# %% [markdown]
# # Proyecto PnP-ADMM para imágenes en baja luminosidad con prior de Deep Denoiser
# 
# A continuación se de# %% [markdown]
# ## Funcion para agregar ruido poisson usando DeepInverse Physics Modelribe un esquema de trabajo para resolver el problema de reconstrucción de una imagen $x$ a partir de observaciones en **baja luz** (ruido de Poisson) combinando un **prior aprendido** (deep denoiser) dentro de un esquema **Plug-and-Play ADMM**.
# 
# ---
# 
# ## 1. Planteamiento del problema
# 
# - **Modelo de medida (Poisson)**  
#   $$
#   y_i \sim \mathrm{Pois}(\gamma\,x_i),
#   \quad i = 1, \dots, m,
#   $$
#   donde $\gamma>0$ es el factor de exposición y $y\in\mathbb N^m$ es la imagen ruidosa vectorizada.
# 
# - **Coste a minimizar** (neg-log-verosimilitud + prior):  
#   $$
#   \min_{x}\;f(x) + g(x),
#   $$
#   con  
#   $$
#   f(x)
#   = \gamma\,\mathbf{1}^T x \;-\; y^T\log(\gamma\,x),
#   \qquad
#   g(x)
#   = \lambda\,R(x),
#   $$
#   donde $R$ es un regularizador implícito definido por un denoiser entrenado.
# 
# ---
# 
# ## 2. Splitting con ADMM
# 
# Introducimos $z$ y la igualdad $x = z$. El problema equivalente es
# 
# $$
# \min_{x,z}\;f(x) + g(z)
# \quad\text{sujeto a}\quad x = z.
# $$
# 
# El Lagrangiano aumentado escalado con $\rho>0$ es
# 
# $$
# \mathcal{L}_\rho(x,z,u)
# = f(x) + g(z)
# + \frac{\rho}{2}\,\|x - z + u\|_2^2.
# $$
# 
# Las iteraciones de ADMM son:
# 
# 1. **Paso $x$**:  
#    $$
#    x^{k+1}
#    = \arg\min_{x}\;
#    f(x)
#    + \frac{\rho}{2}\,\|x - z^k + u^k\|_2^2.
#    $$
# 
# 2. **Paso $z$**:  
#    $$
#    z^{k+1}
#    = \arg\min_z\;
#    g(z)
#    + \frac{\rho}{2}\,\|x^{k+1} - z + u^k\|_2^2
#    \; = 
#    \mathcal{D}(x^{k+1} + u^k),
#    $$
#    donde $\mathcal{D}$ es un denoiser profundo preentrenado.
# 
# 3. **Paso $u$**:  
#    $$
#    u^{k+1}
#    = u^k + (x^{k+1} - z^{k+1}).
#    $$
# 
# ---
# 
# ## 3. Derivada del paso de actualizacion de $x$
# 
# encontrar el valor minimo para el paso de $x$
# 
# $$
#   x^{k+1}
#    = \arg\min_{x}\;
#    f(x)
#    + \frac{\rho}{2}\,\|x - z + u\|_2^2.
# $$
# 
# primero definamos la derivada de valor por valor
# 
# $$
#   \frac{\partial f}{\partial x_i}  = 
#   \frac{\partial}{\partial x_i} \left[
#     \gamma x_i - y_i \log(\gamma x_i) +
#     \frac{\rho}{2}(x_i - z_i + u_i)^2
#   \right]
# $$
# 
# $$
#   \frac{\partial f}{\partial x_i}  = 
#     \gamma - y_i \frac{1}{\gamma x_i} \frac{\partial}{\partial x_i} \left[ \gamma x_i \right] +
#     \frac{\rho}{2}\cdot 2 \cdot(x_i - z_i + u_i)
# $$
# 
# $$
#   \frac{\partial f}{\partial x_i}  = 
#     \gamma - \frac{y_i}{x_i} +
#     \rho(x_i - z_i + u_i)
# $$
# 
# con base en esto podemos hallar el argumento minimo por la primera derivada ($\frac{df}{dx} =0$)
# 
# $$
#     \gamma - \frac{y_i}{x_i} +
#     \rho(x_i - z_i + u_i) = 0
# $$
# 
# $$
#     \gamma x_i - y_i +
#     \rho(x_i^2 - z_i x_i + u_i x_i) = 0
# $$
# 
# $$
#   \rho x_i^2 + (\gamma - \rho z_i + \rho u_i)x_i - y_i = 0
# $$
# 
# 
# ---
# 
# ## 4. Actualización cerrada de $x$ (pixel-a-pixel)
# 
# Buscamos
# 
# $$
# x_i^{k+1}
# = \arg\min_{x_i}\;
# \gamma\,x_i - y_i\log(\gamma\,x_i)
# + \frac{\rho}{2}\,(x_i - z_i + u_i)^2.
# $$
# 
# La condición de óptimo conduce a la ecuación cuadrática
# 
# $$
# \rho\,x_i^2 + (\gamma - \rho\,z_i + \rho\,u_i)\,x_i - y_i = 0,
# $$
# 
# para usar
# 
# $$
# x = \frac{-b + \pm \sqrt{b^2 - 4ac} }{2a}
# $$
# 
# cuyas raíces dan
# 
# $$
# x_i^{k+1}
# = \frac{-(\gamma - \rho z_i + \rho u_i)
#   + \sqrt{(\gamma - \rho z_i + \rho u_i)^2 + 4\,\rho\,y_i}}
#        {2\,\rho},
# $$
# 
# 

# %% [markdown]
# # Librerias

# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
import torch
import deepinv as dinv

# denoiser
from deepinv.models import Restormer

# %% [markdown]
# # Carga de imagen y resize

# %%
size = 512

image = cv2.imread("lena567.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (size, size))
image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

print("image shape:", image.shape)
print("image max:", image.max())
print("image min:", image.min())

# plot
plt.figure(figsize=(8, 6))
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.title('Imagen Original')
plt.savefig('imagen_original.png', dpi=150, bbox_inches='tight')
plt.close()
print("Imagen original guardada en 'imagen_original.png'")

# %% [markdown]
# ## Funcion para agregar ruido poisson a una imagen

# %%
# Physics Model: Ruido Poisson usando DeepInverse
gamma = 10.0
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Crear el modelo de ruido Poisson 
# IMPORTANTE: normalize=False para manejar manualmente la escala
noise_model = dinv.physics.PoissonNoise(
    gain=gamma, 
    normalize=False,  # Manejaremos la normalización manualmente
    clip_positive=True  # Asegurar valores positivos
)

# Crear el physics model usando DeepInverse
physics = dinv.physics.Denoising(noise_model)

# Convertir imagen a tensor para el physics model
# Escalar la imagen para que funcione bien con Poisson (multiplicar por gamma)
x_scaled = image * gamma  # Escalar para el modelo Poisson
x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

print(f"Imagen escalada - min: {x_tensor.min():.4f}, max: {x_tensor.max():.4f}")

# Aplicar el physics model para generar la imagen ruidosa
with torch.no_grad():
    y_tensor_scaled = physics(x_tensor)

# Convertir de vuelta al rango [0,1] dividiendo por gamma
y_tensor = y_tensor_scaled / gamma
y_tensor = torch.clamp(y_tensor, 0.0, 1.0)

# Convertir de vuelta a numpy para visualización
y = y_tensor.squeeze().cpu().numpy()

# %% [markdown]
# ## Visualización de la imagen ruidosa

# %%
# plot noisy image
print("noisy image shape:", y.shape)
print("noisy image max:", y.max())
print("noisy image min:", y.min())

plt.figure(figsize=(8, 6))
plt.imshow(y, cmap="gray")
plt.axis("off")
plt.title('Imagen con Ruido Poisson (DeepInverse)')
plt.savefig('imagen_ruidosa.png', dpi=150, bbox_inches='tight')
plt.close()
print("Imagen ruidosa guardada en 'imagen_ruidosa.png'")

# %% [markdown]
# # Definición de los pasos del algoritmo ADMM

# %% [markdown]
# ## Paso $x$ - Actualización de la imagen

# %%
def paso_x(y, gamma, rho, z, u):
    """
    Paso x-update: Resolver la ecuación cuadrática pixel a pixel
    
    Resuelve: x = argmin[γ*x - y*log(γ*x) + (ρ/2)||x - z + u||²]
    
    La ecuación cuadrática es: ρx² + (γ - ρz + ρu)x - y = 0
    
    :param y: Imagen ruidosa (tensor) - debe estar escalada por gamma
    :param gamma: Parámetro del ruido Poisson
    :param rho: Parámetro de penalización ADMM
    :param z: Variable auxiliar actual
    :param u: Variable dual actual
    :return: x actualizado
    """
    a = rho
    b = gamma - rho * z + rho * u
    c = -y
    
    # Fórmula cuadrática (tomar raíz positiva)
    discriminant = b**2 - 4*a*c
    x_k = (-b + torch.sqrt(torch.clamp(discriminant, min=1e-10))) / (2*a)
    x_k = torch.clamp(x_k, min=1e-8)  # Evitar valores negativos o cero
    
    return x_k

# %% [markdown]
# ## Paso $z$ - Aplicación del denoiser

# %%
def paso_z(x, u, denoiser, device):
    """
    Paso z-update: Aplicar el denoiser como operador proximal
    
    z = D(x + u) donde D es el denoiser
    
    :param x: Imagen actual
    :param u: Variable dual actual
    :param denoiser: Modelo de deep denoiser
    :param device: Dispositivo de cómputo
    :return: z actualizado
    """
    # Combinar x + u
    arg = x + u
    
    # Asegurar que esté en rango válido [0,1] para el denoiser
    arg = torch.clamp(arg, 0.0, 1.0)
    
    # Aplicar denoiser (necesita dimensiones para batch y canal)
    if arg.dim() == 2:
        arg = arg.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        z_k = denoiser(arg)
    
    # Remover dimensiones extra y retornar
    z_k = z_k.squeeze()
    z_k = torch.clamp(z_k, 0.0, 1.0)  # Mantener en [0,1]
    
    return z_k

# %% [markdown]
# ## Paso $u$ - Actualización de multiplicadores

# %%
def paso_u(x, z, u):
    """
    Paso u-update: Actualizar los multiplicadores de Lagrange
    
    u = u + (x - z)
    
    :param x: Imagen actual
    :param z: Variable auxiliar actual
    :param u: Variable dual actual
    :return: u actualizado
    """
    u_k = u + (x - z)
    return u_k

# %% [markdown]
# # Configuración del problema

# %%
# =======================
# Configuración de parámetros para ADMM
# =======================
print(f"Physics model ya configurado. Gamma = {gamma}")

y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Inicialización de variables ADMM
x_init = torch.tensor(y, dtype=torch.float32).to(device)
z_init = torch.tensor(y, dtype=torch.float32).to(device)

# =======================
# Prior: Deep Denoiser como operador proximal
# =======================
print("Cargando modelo Restormer...")
denoiser = Restormer(
    in_channels=1,
    out_channels=1,
    pretrained="denoising_gray",
    device=device,
)
print("Modelo Restormer cargado exitosamente")

# %% [markdown]
# # Implementación del algoritmo ADMM

# %%
# =======================
# Implementación manual de ADMM usando las funciones definidas
# =======================
def admm_manual(y, gamma, denoiser, rho=0.1, max_iter=30, tol=1e-6):
    """
    Implementación manual de ADMM para Poisson + Deep Denoiser
    usando las funciones paso_x, paso_z, paso_u definidas anteriormente
    """
    # Escalar y para el modelo Poisson (multiplicar por gamma)
    y_scaled = y * gamma
    y_tensor = torch.from_numpy(y_scaled).to(device).float()
    
    # Inicialización en el rango [0,1]
    x = torch.tensor(y, dtype=torch.float32, device=device)  # Inicializar sin escalar
    z = torch.tensor(y, dtype=torch.float32, device=device)
    u = torch.zeros_like(x)
    
    print(f"Iniciando ADMM manual con rho={rho}, max_iter={max_iter}")
    print(f"y_scaled range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
    residuals = []
    
    for k in range(max_iter):
        x_old = x.clone()
        
        # x-update usando la función definida (y_tensor ya está escalado)
        x = paso_x(y_tensor, gamma, rho, z, u)
        
        # z-update usando la función definida
        z = paso_z(x, u, denoiser, device)
        
        # u-update usando la función definida
        u = paso_u(x, z, u)
        
        # Calcular residuos para convergencia
        residual_primal = torch.norm(x - z).item()
        residual_dual = rho * torch.norm(x - x_old).item()
        residuals.append((residual_primal, residual_dual))
        
        if k % 5 == 0:
            # Calcular el costo con valores escalados correctamente
            x_scaled_for_cost = x * gamma
            cost = torch.sum(x_scaled_for_cost - y_tensor * torch.log(x_scaled_for_cost + 1e-10)).item()
            print(f"Iter {k}: Residual primal = {residual_primal:.6f}, "
                  f"dual = {residual_dual:.6f}, Costo = {cost:.4f}")
        
        # Criterio de convergencia
        if residual_primal < tol and residual_dual < tol:
            print(f"Convergencia alcanzada en iteración {k}")
            break
    
        return x.detach().cpu().numpy(), residuals

# %% [markdown]
# # Ejecución del algoritmo y resultados

# %%
# =======================
# Ejecutar optimización manual
# =======================
print("Ejecutando ADMM manual...")
x_restaurada, residuals = admm_manual(y, gamma=gamma, denoiser=denoiser, rho=0.1, max_iter=300)

print(f"Imagen restaurada - shape: {x_restaurada.shape}")
print(f"Imagen restaurada - min: {x_restaurada.min():.4f}, max: {x_restaurada.max():.4f}")

# %% [markdown]
# # Visualización y métricas de resultados

# %%
# =======================
# Mostrar resultados
# =======================
 # Visualizar
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(image, cmap='gray')
axes[0,0].set_title('Original')
axes[0,0].axis('off')

axes[0,1].imshow(y, cmap='gray')
axes[0,1].set_title('Ruidosa')
axes[0,1].axis('off')

axes[0,2].imshow(x_restaurada, cmap='gray')
axes[0,2].set_title('Restaurada (Corregida)')
axes[0,2].axis('off')

# Gráficas de convergencia
if residuals:
    primal_res = [r[0] for r in residuals]
    dual_res = [r[1] for r in residuals]
    
    axes[1,0].semilogy(primal_res, 'b-', label='Primal')
    axes[1,0].semilogy(dual_res, 'r-', label='Dual')
    axes[1,0].set_title('Convergencia')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Normalizar la imagen restaurada antes de calcular el error
    x_restaurada_norm = (x_restaurada - x_restaurada.min()) / (x_restaurada.max() - x_restaurada.min())
    
    axes[1,1].imshow(np.abs(image - x_restaurada_norm), cmap='hot')
    axes[1,1].set_title('Error Absoluto (Normalizado)')
    axes[1,1].axis('off')
    
    axes[1,2].hist(x_restaurada.flatten(), bins=50, alpha=0.7)
    axes[1,2].set_title('Histograma Restaurada')

plt.tight_layout()
plt.savefig('resultados_corregidos.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# =======================
# Calcular métricas de calidad
# =======================
def calculate_psnr(img1, img2):
    """Calcular PSNR entre dos imágenes"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

# Normalizar la imagen restaurada para métricas justas
x_restaurada_norm = (x_restaurada - x_restaurada.min()) / (x_restaurada.max() - x_restaurada.min())

psnr_noisy = calculate_psnr(image, y)
psnr_restored = calculate_psnr(image, x_restaurada_norm)

print("\n" + "="*50)
print("MÉTRICAS DE CALIDAD:")
print("="*50)
print(f"PSNR imagen ruidosa: {psnr_noisy:.2f} dB")
print(f"PSNR imagen restaurada: {psnr_restored:.2f} dB")
print(f"Mejora en PSNR: {psnr_restored - psnr_noisy:.2f} dB")
print("="*50)

# %% [markdown]
# # Resultados y análisis

print("Ejecución completada. Los resultados han sido guardados en archivos de imagen.")


