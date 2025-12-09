import os
# --- OPTIMIZACIÓN: Silenciar advertencias de TensorFlow ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = Mostrar solo errores
# --------------------------------------------------------

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 🚨 NOTA: Se requiere la clase 'Foto' definida, asumiremos que está en 'fotografia.py'
from fotografia import Foto 

# ===========================
#   CONFIGURACIÓN DEL MODELO FINAL (V4: 256x256)
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. RUTA DEL MODELO
MODEL_NAME = "model_Digits_2.keras"
# Asumo que la carpeta 'models' está dentro de BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", MODEL_NAME) # Ruta ajustada a 'src/models'

# 2. TAMAÑO DE ENTRADA
INPUT_SIZE = (256, 256)

# 3. MAPA DE CLASES (TRADUCCIÓN/CORRECCIÓN FINAL)
# Este es el array que usamos para la corrección final:
# (Asegúrate de que este sea el mapeo final que quieres usar)
DIGITS_CORRECTED = ['0', '1', '2', '3', '4', '8', '5', '9', '7', '6'] 

# El array de mapeo que se usará para traducir el índice (0-9) a tu dígito final
INDEX_TO_DIGIT = {i: digit for i, digit in enumerate(DIGITS_CORRECTED)}

DATA_DIR = os.path.join(BASE_DIR, "src", "data", "predict")


# Cargar modelo al iniciar el backend
try:
    model = load_model(MODEL_PATH, compile=False) 
    print(f"Modelo {MODEL_NAME} cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo en: {MODEL_PATH}")
    print(f"Detalle del error: {e}")
    exit()


def prediction(nombre_foto: str) -> Foto:
    """
    Realiza la predicción, almacena las probabilidades y el resultado corregido
    en el objeto Foto.
    """

    ruta_imagen = os.path.join(DATA_DIR, nombre_foto)

    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    foto = Foto()

    # --- PREPROCESAMIENTO ---
    img = Image.open(ruta_imagen).convert("L")

    if img.size != INPUT_SIZE:
        img = img.resize(INPUT_SIZE)

    img_arr = np.array(img, dtype=np.float32) / 255.0
    foto.set_size(img_arr.shape)

    img_arr = np.expand_dims(img_arr, axis=[0, -1]) 

    # --- PREDICCIÓN ---
    pred = model.predict(img_arr, verbose=0)
    probabilities = pred[0]
    
    # 🚨 ALMACENAR PROBABILIDADES en el objeto Foto
    foto.set_probabilities(probabilities.tolist()) 
    
    index = np.argmax(probabilities)
    predicted_digit = INDEX_TO_DIGIT[index]
    
    # Imprime el diagnóstico
    print(f"\n--- DIAGNÓSTICO para {nombre_foto} ---")
    
    top_indices = np.argsort(probabilities)[::-1]
    
    print(f"Predicción (Índice): {index}")
    print(f"Predicción (Dígito Corregido): {predicted_digit}")
    print("\nPROBABILIDADES DETALLADAS (Índice: Dígito Corregido):")

    for i in top_indices:
        if i < 5 or probabilities[i] * 100 > 0.001:
            # Usamos INDEX_TO_DIGIT para mostrar el dígito corregido
            print(f"  {INDEX_TO_DIGIT[i]} (Índice {i}): {probabilities[i]*100:.2f}%")
        else:
            break

    # Almacenar el resultado corregido
    foto.set_predicted_label(predicted_digit)

    return foto

def run_all_tests():
    """Ejecuta la predicción para foto_0.jpg a foto_9.jpg."""
    print("\n===========================================")
    print("EJECUTANDO PRUEBAS CON MAPEO CORREGIDO")
    print("===========================================")
    
    for i in range(10):
        img_test = f"foto_{i}.jpg"
        try:
            foto = prediction(img_test)
            # Usamos el getter corregido
            print(f"RESULTADO FINAL ESPERADO ({i}): {foto.get_predicted_label()}") 
        except FileNotFoundError:
            print(f"\nADVERTENCIA: Archivo {img_test} no encontrado. Saltando.")
        except Exception as e:
             # El error de atributo está corregido en la lógica de impresión
             print(f"\nERROR general durante la predicción de {img_test}: {e}")

# ===========================
#   INICIO DEL SCRIPT
# ===========================
if __name__ == "__main__":
    run_all_tests()