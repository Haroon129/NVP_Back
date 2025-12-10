import cloudinary
import cloudinary.uploader
import io
from typing import Optional
import os
from dotenv import load_dotenv # 🆕 Importar load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

# --- Configuración de Cloudinary (Usando variables de entorno) ---
cloudinary.config( 
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.environ.get("CLOUDINARY_API_KEY"),        
    api_secret = os.environ.get("CLOUDINARY_API_SECRET"),  
    secure=True
)

def subir_imagen_a_cloudinary(nombre_publico: str, contenido_bytes: bytes) -> Optional[str]:
    """
    Sube el contenido binario de una imagen a Cloudinary y devuelve la URL segura.

    Args:
        nombre_publico: El ID público que tendrá la imagen en Cloudinary (ej: 'mano_123').
        contenido_bytes: El contenido binario (bytes) de la imagen.

    Returns:
        La URL HTTPS segura del archivo subido o None si falla.
    """
    try:
        # Usamos io.BytesIO para manejar los bytes en memoria
        file_obj = io.BytesIO(contenido_bytes)
        
        # Realizar la subida
        upload_result = cloudinary.uploader.upload(
            file_obj,
            resource_type='image',
            public_id=nombre_publico,
            folder="NVP_Images" 
        )
        
        # Cloudinary devuelve la URL segura inmediatamente
        return upload_result.get("secure_url")

    except Exception as error:
        print(f"❌ Ocurrió un error al subir a Cloudinary: {error}")
        return None