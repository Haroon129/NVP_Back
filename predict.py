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
MODEL_NAME = "model_Digits_2.keras" # Asegúrate de que este nombre sea correcto
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

# 2. TAMAÑO DE ENTRADA
INPUT_SIZE = (256, 256)

# 3. MAPA DE CLASES (TRADUCCIÓN/CORRECCIÓN FINAL)
# Nota: Este es el mapeo de ejemplo que corrige los errores más limpios (4 y 5/6)
DIGITS              = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
DIGITS_CORRECTED    = ['0', '1', '2', '3', '4', '8', '5', '9', '7', '6'] 
INDEX_TO_DIGIT = {i: digit for i, digit in enumerate(DIGITS)}

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
    # ... (código de preprocesamiento, predicción y diagnóstico sin cambios) ...

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

    # --- PREDICCIÓN Y DIAGNÓSTICO ---
    pred = model.predict(img_arr, verbose=0)
    probabilities = pred[0]
    
    index = np.argmax(probabilities)
    predicted_digit = INDEX_TO_DIGIT[index]
    
    # Imprime las probabilidades completas para diagnóstico
    print(f"\n--- DIAGNÓSTICO para {nombre_foto} ---")
    
    top_indices = np.argsort(probabilities)[::-1]
    
    print(f"Predicción (Índice): {index}")
    print(f"Predicción (Dígito Corregido): {predicted_digit}")
    print("\nPROBABILIDADES DETALLADAS:")

    for i in top_indices:
        if i < 5 or probabilities[i] > 0.001:
            # Mostramos el dígito corregido
            print(f"  Dígito {INDEX_TO_DIGIT[i]} (Índice {i}): {probabilities[i]*100:.2f}%")
        else:
            break

    # 4. Guardar la predicción final
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
            # 🚨 CORRECCIÓN DE SINTAXIS AQUÍ
            print(f"RESULTADO FINAL ESPERADO ({i}): {foto.get_predicted_label()}") 
        except FileNotFoundError:
            print(f"\nADVERTENCIA: Archivo {img_test} no encontrado. Saltando.")
        except AttributeError:
             print(f"\nERROR: Atributo incorrecto. Usando .get_predicted_label() o atributo incorrecto.")
        except Exception as e:
             print(f"\nERROR general durante la predicción de {img_test}: {e}")

# ===========================
#   INICIO DEL SCRIPT
# ===========================
if __name__ == "__main__":
    run_all_tests()