# Resumen Ejecutivo del Proyecto

## PnP-ADMM para Restauración de Imágenes en Baja Luminosidad

**Autores**: Miguel Fernando Pimiento Escobar, Jorge Andrey Gracia Vanegas  
**Curso**: Optimización Convexa  
**Fecha**: Junio 2025

---

## 🎯 Objetivo del Proyecto

Desarrollar e implementar un algoritmo **Plug-and-Play ADMM** que resuelva el problema de restauración de imágenes degradadas por **ruido Poisson** en condiciones de baja luminosidad, utilizando un **Deep Denoiser** como prior implícito.

## 📊 Problema Matemático

### Formulación
Resolver el problema de optimización no-convexa:

$$\min_{x \geq 0} \left\{ \underbrace{\gamma \mathbf{1}^T x - y^T \log(\gamma x)}_{\text{Fidelidad Poisson}} + \underbrace{\lambda R(x)}_{\text{Prior implícito}} \right\}$$

### Estrategia de Solución
1. **Variable auxiliar**: $x = z$ para separar términos difíciles
2. **ADMM**: Algoritmo iterativo que alterna entre subproblemas más simples
3. **Plug-and-Play**: Reemplazar operador proximal por denoiser preentrenado

## 🔧 Contribuciones Técnicas

### 1. Derivación Matemática Completa
- **Ecuación cuadrática** para el x-update: $\rho x^2 + (\gamma - \rho z + \rho u)x - y = 0$
- **Solución analítica** píxel a píxel usando fórmula cuadrática
- **Análisis de convergencia** mediante residuos primal y dual

### 2. Implementación Robusta
- **Estabilidad numérica**: Manejo de casos límite y overflow
- **Escalado apropiado**: Normalización correcta para modelo Poisson
- **Integración con DeepInverse**: Uso de physics models estándar

### 3. Arquitectura Modular
- **8 módulos especializados**: Separación clara de responsabilidades
- **API limpia**: Interfaz simple para uso y extensión
- **Documentación completa**: Matemática, técnica y de uso

## 📈 Resultados Obtenidos

### Métricas de Calidad
- **PSNR**: Mejoras típicas de +5 a +15 dB
- **Convergencia**: Estable en 30-50 iteraciones
- **Tiempo**: ~10-30 segundos en GPU moderna
- **Calidad visual**: Preservación de detalles y reducción efectiva de ruido

### Casos de Prueba
- **Imagen Lena 512×512**: Restauración exitosa con γ=10.0
- **Diferentes niveles de ruido**: Robusto para γ ∈ [5, 50]
- **Convergencia garantizada**: En todos los casos probados

## 📋 Documentación Completa

### Documentos Matemáticos
1. **[SOLUCION_MATEMATICA.md](SOLUCION_MATEMATICA.md)**
   - Formulación rigorosa del problema
   - Derivación completa del algoritmo ADMM
   - Análisis teórico de convergencia
   - Propiedades matemáticas

2. **[IMPLEMENTACION_TECNICA.md](IMPLEMENTACION_TECNICA.md)**
   - Derivación paso a paso del x-update
   - Consideraciones de estabilidad numérica
   - Optimización de rendimiento
   - Debugging y diagnóstico

### Documentos de Código
3. **[ESTRUCTURA_MODULAR.md](ESTRUCTURA_MODULAR.md)**
   - Organización del código en módulos
   - API de uso y ejemplos
   - Ventajas de la modularización

4. **[README.md](README.md)**
   - Guía de uso general
   - Instalación y dependencias
   - Ejemplos de uso

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Imagen        │    │  Ruido Poisson  │    │   Algoritmo     │
│   Original      │───▶│  y~Pois(γx)     │───▶│   PnP-ADMM      │
│   x ∈ [0,1]     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Imagen        │    │  Deep Denoiser  │    │   Iteraciones   │
│   Restaurada    │◀───│  (Restormer)    │◀───│   x,z,u steps   │
│   x̂ ∈ [0,1]     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```
