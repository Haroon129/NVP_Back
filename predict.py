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
#    CONFIGURACIÓN INICIAL
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. RUTA DEL MODELO
MODEL_NAME = "model.keras"
MODEL_PATH = os.path.join(BASE_DIR, "src","models",MODEL_NAME) 

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
    Realiza la predicción, almacena las probabilidades y el resultado corregido
    en el objeto Foto.
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
    probabilities = pred[0]
    
     # 🚨 ALMACENAR PROBABILIDADES en el objeto Foto
    foto.set_probabilities(probabilities.tolist()) 
    
    index = np.argmax(probabilities)
    predicted_class = INDEX_TO_CLASS[index]
    
     # Imprime el diagnóstico
    print(f"\n--- DIAGNÓSTICO para {nombre_foto} ---")
    
    top_indices = np.argsort(probabilities)[::-1]
    
    print(f"Predicción (Índice): {index}")
    print(f"Predicción (Clase ASL): {predicted_class}")
    print("\nPROBABILIDADES DETALLADAS (Clase ASL):")

     # Imprimir las 5 mejores predicciones
    for rank, i in enumerate(top_indices[:5]):
         # Usamos INDEX_TO_CLASS para mostrar la clase ASL
        print(f" {rank+1}. {INDEX_TO_CLASS[i]} (Confianza): {probabilities[i]*100:.2f}%")

     # Almacenar el resultado corregido
    foto.set_predicted_label(predicted_class)

    return foto

def run_all_tests():
    """Ejecuta la predicción para las 29 imágenes del Test Set."""
    if model is None:
        return

    print("\n===========================================")
    print(f"EJECUTANDO PRUEBAS EN EL TEST SET (N={NUM_CLASSES} Clases)")
    print("===========================================")
    
     # Nombres base de las 29 clases
    base_labels = list(string.ascii_uppercase) + ['delete', 'nothing', 'space']
    
     # Archivos de prueba esperados (Ej: A_test.jpg, delete_test.jpg)
    test_files = [f"{label}_test.jpg" for label in base_labels]
    
    correct_count = 0
    total_tests = 0

    for img_test in test_files:
        try:
            foto = prediction(img_test)
    
            # Extraer la etiqueta verdadera (Ej: 'A' de 'A_test.jpg')
            true_label = img_test.split('_')[0]
            predicted_label = foto.get_predicted_label().split(' ')[0]

            is_correct = "✅ CORRECTO" if true_label == predicted_label else "❌ FALLO"
    
            if true_label == predicted_label:
              correct_count += 1
    
            total_tests += 1

            print(f"--- RESULTADO {is_correct} ({img_test}) ---")
            print(f"Etiqueta Real: {true_label}")
            print(f"Predicción: {foto.get_predicted_label()}")
    
        except FileNotFoundError:
            # Esto es esperado si el Test Set no está completo
            continue # Ignoramos la advertencia para no saturar la salida
        except Exception as e:
            print(f"\nERROR general durante la predicción de {img_test}: {e}")
    
    if total_tests > 0:
         accuracy = (correct_count / total_tests) * 100
         print("\n===========================================")
         print(f"📊 Resumen de Precisión ({total_tests} Pruebas): {accuracy:.2f}%")
         print("===========================================")


"""
img = "foto_U.png"
foto = prediction(img)

print(f"Predicción: {foto.get_predicted_label()}")"""