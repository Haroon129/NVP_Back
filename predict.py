import os
# --- Silenciar advertencias de TensorFlow ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# --------------------------------------------

import numpy as np
import string
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps 
from tensorflow.keras.applications.efficientnet import preprocess_input 


from fotografia import Foto 

# ===========================
#    CONFIGURACIÓN INICIAL
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. RUTA DEL MODELO
MODEL_NAME = "model.keras"
MODEL_PATH = os.path.join(BASE_DIR,"models",MODEL_NAME) 

# 2. TAMAÑO DE ENTRADA
INPUT_SIZE = (224, 224) 

# 3. MAPA DE CLASES (29 CLASES)
LABELS = list(string.ascii_uppercase) + ['delete', 'nothing', 'space']
INDEX_TO_CLASS = {i: label for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# 4. RUTA DE LAS FOTOS A PREDECIR
DATA_DIR = os.path.join(BASE_DIR, "src", "data","predict")


# Cargar modelo al iniciar el backend
try:
    # Usamos compile=False ya que solo haremos inferencia
    model = load_model(MODEL_PATH, compile=False) 
    print(f"Modelo {MODEL_NAME} cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo en: {MODEL_PATH}")
    print(f"Detalle del error: {e}")
    # Salida limpia, aunque en un script real podrías usar exit()
    model = None 


def prediction(nombre_foto: str) -> Foto:
    """
    Realiza la predicción, almacena las probabilidades como diccionario 
    y el resultado corregido en el objeto Foto.
    """
    if model is None:
        raise Exception("El modelo no está cargado.")

    ruta_imagen = os.path.join(DATA_DIR, nombre_foto)

    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    foto = Foto()

    img = Image.open(ruta_imagen).convert("RGB") 
    target_tuple = INPUT_SIZE
    try:
        img = ImageOps.fit(img, target_tuple, Image.Resampling.LANCZOS)
    except AttributeError:
        img = ImageOps.fit(img, target_tuple, Image.LANCZOS)

    img_arr = np.array(img, dtype=np.float32)
    foto.set_size(img_arr.shape) 

    # Aplicar la normalización correcta de EfficientNet y añadir el batch dimension
    img_arr = preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)

    # --- PREDICCIÓN ---
    pred = model.predict(img_arr, verbose=0)
    probabilities_array = pred[0]
    
    # Creamos un diccionario donde la clave es la clase (Ej: 'A', 'B') 
    # y el valor es la probabilidad (float).
    probabilities_dict = {
        INDEX_TO_CLASS[i]: float(probabilities_array[i])
        for i in range(NUM_CLASSES)
    }

    # ALMACENAR PROBABILIDADES en el objeto Foto
    foto.set_probabilities(probabilities_dict) 
    
    index = np.argmax(probabilities_array)
    predicted_class = INDEX_TO_CLASS[index]

    foto.set_predicted_label(predicted_class)

    return foto