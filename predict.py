import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from fotografia import Foto

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "predict")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_final.h5")

# Cargar modelo al iniciar el backend
model = load_model(MODEL_PATH)

# Mapa de índices
LETRAS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'] 
INDEX_TO_LETTER = {i: letra for i, letra in enumerate(LETRAS)}


def prediction(nombre_foto: str) -> Foto:
    """
    Recibe la ruta de una imagen,
    crea el objeto Foto,
    ajusta tamaño si es necesario,
    hace el predict
    y devuelve el objeto.
    """

    ruta_imagen = os.path.join(DATA_DIR, nombre_foto)

    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    foto = Foto()

    # Cargar imagen a escala de grises
    img = Image.open(ruta_imagen).convert("L")

    # Si no es 28x28 → redimensionar
    if img.size != (28, 28):
        img = img.resize((28, 28))

    # Convertir a array
    img_arr = np.array(img)

    # Guardar el tamaño usado realmente
    foto.set_size(img_arr.shape)

    # Normalizar
    img_arr = img_arr / 255.0

    # Preparar batch para el modelo CNN
    img_arr = img_arr.reshape(1, 28, 28, 1)

    # Predicción
    pred = model.predict(img_arr)
    index = np.argmax(pred)
    letra = INDEX_TO_LETTER[index]

    foto.set_predicted_label(letra)

    return foto
