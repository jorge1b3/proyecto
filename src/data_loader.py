"""
Carga y procesamiento de imágenes
"""

import numpy as np
import cv2
from config import IMAGE_SIZE

def load_and_preprocess_image(image_path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Cargar y preprocesar una imagen
    
    Args:
        image_path: Ruta a la imagen
        size: Tamaño para redimensionar (size x size)
        
    Returns:
        Imagen normalizada en rango [0,1] como array numpy
    """
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir a escala de grises
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar
    image = cv2.resize(image, (size, size))
    
    # Normalizar a [0, 1]
    image = image.astype(np.float32) / 255.0
    
    print(f"Imagen cargada - shape: {image.shape}, min: {image.min():.4f}, max: {image.max():.4f}")
    
    return image

def validate_image_range(image: np.ndarray, name: str = "imagen") -> None:
    """
    Validar que la imagen esté en el rango esperado [0,1]
    
    Args:
        image: Array de imagen
        name: Nombre descriptivo para mensajes
    """
    min_val, max_val = image.min(), image.max()
    print(f"{name} - min: {min_val:.4f}, max: {max_val:.4f}")
    
    if min_val < 0 or max_val > 1:
        print(f"ADVERTENCIA: {name} fuera del rango [0,1]")
    
    return min_val, max_val
