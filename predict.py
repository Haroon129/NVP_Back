import os
# --- OPTIMIZACIÓN: Silenciar advertencias de TensorFlow ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = Mostrar solo errores
# --------------------------------------------------------

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 🚨 NOTA: Se requiere la clase 'Foto' definida, asumiremos que está en 'fotografia.py'
# Si la clase 'Foto' no existe o está definida en otro lugar, deberás ajustarla.
from fotografia import Foto 

# ===========================
#   CONFIGURACIÓN DEL MODELO ACTUAL
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. ACTUALIZAR RUTA DEL MODELO (Basado en el guardado final)
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "model_digits.h5") 

# 2. TAMAÑO DE ENTRADA (Debe coincidir con el tamaño usado en el entrenamiento)
INPUT_SIZE = (128, 128)

# 3. ACTUALIZAR MAPA DE CLASES (Clases 0-9)
DIGITS = ['0','1','2','3','4','5','6','7','8','9'] 
INDEX_TO_DIGIT = {i: digit for i, digit in enumerate(DIGITS)} # Mapeo de índice a dígito (string)

DATA_DIR = os.path.join(BASE_DIR, "src", "data", "predict")


# Cargar modelo al iniciar el backend
try:
    model = load_model(MODEL_PATH)
    print(f"Modelo {os.path.basename(MODEL_PATH)} cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo en: {MODEL_PATH}")
    print(f"Detalle del error: {e}")
    # Si el modelo no se carga, el script debe terminar
    exit()


def prediction(nombre_foto: str) -> Foto:
    """
    Recibe el nombre de una imagen, realiza el preprocesamiento (reescalado a 128x128, 
    escala de grises, normalización) y ejecuta la predicción.
    """

    ruta_imagen = os.path.join(DATA_DIR, nombre_foto)

    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    foto = Foto()

    # 1. Cargar imagen y convertir a escala de grises ("L" mode)
    img = Image.open(ruta_imagen).convert("L")

    # 2. Redimensionar al tamaño de entrada del modelo (128x128)
    if img.size != INPUT_SIZE:
        img = img.resize(INPUT_SIZE)

    # 3. Convertir a array y Normalizar (dividir por 255.0)
    img_arr = np.array(img, dtype=np.float32) / 255.0

    # Guardar el tamaño usado realmente
    foto.set_size(img_arr.shape) # Esto será (128, 128)

    # 4. Preparar batch para el modelo CNN: (H, W) -> (1, H, W, 1)
    # H=128, W=128, 1 canal (escala de grises)
    img_arr = np.expand_dims(img_arr, axis=[0, -1]) 

    # Predicción
    pred = model.predict(img_arr, verbose=0)
    
    # 5. Obtener el índice con la mayor probabilidad
    index = np.argmax(pred)
    
    # 6. Mapear el índice al dígito correcto (0-9)
    predicted_digit = INDEX_TO_DIGIT[index]

    foto.set_predicted_label(predicted_digit)

    return foto

# --- Ejemplo de uso ---
# Asegúrate de que este archivo exista en 'src/data/predict/foto_1.png'
img_test = "foto_6.png" 
print("\n--- INICIANDO PRUEBA ---")
try:
    foto = prediction(img_test)
    print(f"Predicción finalizada para {img_test}.")
    print(f"Clase predicha (Dígito): {foto.get_predicted_label()}")
except FileNotFoundError as e:
    print(e)
except NameError as e:
    print(f"\nError: {e}. Asegúrate de que la clase 'Foto' esté correctamente definida e importada.")

# ----------------------