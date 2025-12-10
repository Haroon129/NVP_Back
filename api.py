from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import base64
import re
import traceback
import os

from predict import prediction
from database_class import DatabaseConnection  # Clase de conexión a DB
from cloudinari import subir_imagen_a_cloudinary
from dotenv import load_dotenv # 🆕 Importar load_dotenv


app = Flask(__name__)

ROOT = Path(__file__).resolve().parent

# Directorios y configuración
PREDICT_DIR = ROOT / "data" / "predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGE_PATH = PREDICT_DIR / "test.jpg"



# Cargar variables de entorno del archivo .env
load_dotenv() 
# 🆕 Configuración de la Base de Datos (usando os.environ)
DB_CONFIG = {
    'host': os.environ.get("DB_HOST"), 
    'user': os.environ.get("DB_USER"), 
    'database': os.environ.get("DB_DATABASE"), 
    'password': os.environ.get("DB_PASSWORD", ""), # Usamos "" como default si no se encuentra
}

# =========================================================
# FUNCIONES AUXILIARES (RESTAURADAS)
# =========================================================

def to_jsonable(obj):
    # Lógica de conversión que definiste (ej. Path a str, numpy a int/float)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode("utf-8")

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    
    # ... (Otras lógicas de conversión si las tenías) ...
    
    return str(obj)


def error_response(status_code: int, error: str, request_id: str, details=None, extra=None):
    # Lógica de respuesta de error que definiste
    payload = {
        "error": error,
        "request_id": request_id,
    }
    if details is not None:
        payload["details"] = details
    if extra is not None:
        payload["extra"] = extra
    return jsonify(payload), status_code

# =========================================================
# ENDPOINT /PREDICT (MODIFICADO Y CORREGIDO)
# =========================================================

@app.route("/predict", methods=["POST"])
def hand_sign_predict():
    request_id = uuid.uuid4().hex[:10]
    
    saved_url = None
    real_label = -1
    img_bytes = None
    file_mime_type = 'image/jpeg' 

    try:
        nombre = None
        unique_name = None

        if request.files and "imagen" in request.files:
            # --- Proceso con Archivo (multipart/form-data) ---
            file = request.files.get("imagen")
            nombre = (request.form.get("nombre") or "").strip()
            
            try:
                real_label = int(request.form.get("label", -1))
            except (ValueError, TypeError):
                pass
            file_mime_type = file.mimetype or file_mime_type
            
            if not nombre:
                return error_response(400, 'Falta el campo "nombre".', request_id)
            if file is None or not file.filename:
                return error_response(400, 'Falta el archivo "imagen".', request_id)

            original_filename = secure_filename(file.filename)
            ext = Path(original_filename).suffix.lower() or ".jpg"
            unique_name = f"{secure_filename(nombre)}_{uuid.uuid4().hex}{ext}"
            
            file.seek(0)
            img_bytes = file.read()

        else:
            # --- Proceso con JSON (Base64) ---
            data = request.get_json(silent=True) or {}
            nombre = (data.get("nombre") or "").strip()
            foto_b64 = (data.get("imagen") or "").strip()
            
            try:
                real_label = int(data.get("label", -1))
            except (ValueError, TypeError):
                pass

            if not nombre or not foto_b64:
                return error_response(
                    400, 'Faltan "nombre" y/o "imagen".', request_id, 
                    details={"got_keys": list(data.keys())}
                )

            foto_b64 = re.sub(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", "", foto_b64)
            
            try:
                img_bytes = base64.b64decode(foto_b64, validate=True)
            except Exception as e:
                return error_response(400, "Base64 inválido en 'imagen'.", request_id, details=str(e))
            
            unique_name = f"{secure_filename(nombre)}_{uuid.uuid4().hex}.jpg"
            
        # 2. Guardar Temporalmente la Imagen para la Predicción
        if img_bytes is None:
             return error_response(500, "No se pudo obtener el contenido de la imagen.", request_id)
             
        TEST_IMAGE_PATH.write_bytes(img_bytes)

        public_id_name = Path(unique_name).stem 
        saved_url = subir_imagen_a_cloudinary(public_id_name, img_bytes)
        
        if saved_url is None:
            return error_response(
                500,
                "Error al subir la imagen a Cloudinary.",
                request_id
            )

        # 4. EJECUTAR PREDICCIÓN
        pred_result = None
        last_err = None
        
        for arg in (str(TEST_IMAGE_PATH), TEST_IMAGE_PATH.name, "test.jpg", nombre):
            try:
                pred_result, top_indices = prediction(arg) 
                last_err = None
                break
            except Exception as e:
                last_err = e

        if pred_result is None:
            print(f"[{request_id}] prediction() falló. Last error: {last_err}")
            print(traceback.format_exc())
            return error_response(500, "Error ejecutando prediction()", request_id, details=str(last_err))
            
        predicted_label = pred_result.get_predicted_label()
        
        # 5. GUARDAR EN LA BASE DE DATOS (NVP.imagenes)
        db_conn = DatabaseConnection(**DB_CONFIG)
        
        insert_sql = """
        INSERT INTO imagenes (URL, predicted_label, label) 
        VALUES (%s, %s, %s)
        """
        insert_params = (saved_url, predicted_label, real_label)
        
        db_success = False
        with db_conn as db:
            if db.cnx:
                db_success = db.execute_query(insert_sql, insert_params)
        
        # 6. LIMPIEZA: Eliminar el archivo temporal
        if TEST_IMAGE_PATH.exists():
            os.remove(TEST_IMAGE_PATH)

        # 7. RESPUESTA FINAL

        meta = {
            "image_url": saved_url,
            "db_inserted": db_success
        }
        
        return jsonify({"prediccion": predicted_label, "meta": meta, "top_indices": top_indices})

    except Exception as e:
        print(f"[{request_id}] ERROR en /predict: {e}")
        print(traceback.format_exc())
        return error_response(500, "Error procesando imagen en backend", request_id, details=str(e))
        


@app.route('/predict/correcta', methods=["POST"])
def pred_correcta():
    url = request.form.get('url')
    db = DatabaseConnection()
    return
       
@app.route('/predict/incorrecta',methods=["POST"])
def pred_incorrecta():
    return



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), debug=True)