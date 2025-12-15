from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pathlib import Path
import uuid
import base64
import re
import traceback
import os

from predict import prediction
from database_class import DatabaseConnection
from cloudinari import subir_imagen_a_cloudinary
from dotenv import load_dotenv

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": ["https://nvp.heradome.com"]}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


ROOT = Path(__file__).resolve().parent

# Directories and configuration
PREDICT_DIR = ROOT / "data" / "predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGE_PATH = PREDICT_DIR / "test.jpg"

# Load environment variables from the .env file
load_dotenv()

def _db_config():
    """
    En Docker + MySQL local (XAMPP en tu host):
    - Mac/Windows: host.docker.internal
    - Linux: tendrás que usar extra_hosts o la IP del host.
    """
    return {
        "host": os.environ.get("DB_HOST") or "host.docker.internal",
        "user": os.environ.get("DB_USER") or "root",
        "database": os.environ.get("DB_DATABASE") or "NVP",
        "password": os.environ.get("DB_PASSWORD", ""),
    }

DB_CONFIG = _db_config()

def db_config_sanitized(cfg: dict) -> dict:
    safe = dict(cfg or {})
    if "password" in safe:
        safe["password"] = "***"
    return safe

# =========================================================
# AUXILIARY FUNCTIONS
# =========================================================

def to_jsonable(obj):
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

    return str(obj)


def error_response(status_code: int, error: str, request_id: str, details=None, extra=None):
    payload = {"error": error, "request_id": request_id}
    if details is not None:
        payload["details"] = to_jsonable(details)
    if extra is not None:
        payload["extra"] = to_jsonable(extra)
    return jsonify(payload), status_code


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return jsonify(ok=True, service="nvp-back", endpoints=["/predict", "/predict/correcta", "/predict/incorrecta"]), 200


@app.route("/predict", methods=["POST"])
def hand_sign_predict():
    request_id = uuid.uuid4().hex[:10]

    saved_url = None
    real_label = -1
    img_bytes = None

    try:
        nombre = None
        unique_name = None

        # --- 1) HANDLE INPUT (File or Base64) ---
        if request.files and "imagen" in request.files:
            file = request.files.get("imagen")
            nombre = (request.form.get("nombre") or "").strip()

            try:
                real_label = int(request.form.get("label", -1))
            except (ValueError, TypeError):
                pass

            if not nombre:
                return error_response(400, 'Missing field "nombre".', request_id)
            if file is None or not file.filename:
                return error_response(400, 'Missing file "imagen".', request_id)

            original_filename = secure_filename(file.filename)
            ext = Path(original_filename).suffix.lower() or ".jpg"
            unique_name = f"{secure_filename(nombre)}_{uuid.uuid4().hex}{ext}"

            file.seek(0)
            img_bytes = file.read()

        else:
            data = request.get_json(silent=True) or {}
            nombre = (data.get("nombre") or "").strip()
            foto_b64 = (data.get("imagen") or "").strip()

            try:
                real_label = int(data.get("label", -1))
            except (ValueError, TypeError):
                pass

            if not nombre or not foto_b64:
                return error_response(
                    400,
                    'Missing "nombre" and/or "imagen".',
                    request_id,
                    details={"got_keys": list(data.keys())},
                )

            foto_b64 = re.sub(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", "", foto_b64)

            try:
                img_bytes = base64.b64decode(foto_b64, validate=True)
            except Exception as e:
                return error_response(400, "Invalid Base64 in 'imagen'.", request_id, details=str(e))

            unique_name = f"{secure_filename(nombre)}_{uuid.uuid4().hex}.jpg"

        # --- 2) Save temp image ---
        if img_bytes is None:
            return error_response(500, "Could not retrieve image content.", request_id)

        TEST_IMAGE_PATH.write_bytes(img_bytes)

        # --- 3) Upload to Cloudinary ---
        public_id_name = Path(unique_name).stem
        saved_url = subir_imagen_a_cloudinary(public_id_name, img_bytes)

        if saved_url is None:
            return error_response(500, "Error uploading image to Cloudinary.", request_id)

        # --- 4) Prediction ---
        pred_result = None
        last_err = None

        for arg in (str(TEST_IMAGE_PATH), TEST_IMAGE_PATH.name, "test.jpg", nombre):
            try:
                pred_result = prediction(arg)
                last_err = None
                break
            except Exception as e:
                last_err = e

        if pred_result is None:
            print(f"[{request_id}] prediction() failed. Last error: {last_err}")
            print(traceback.format_exc())
            return error_response(500, "Error executing prediction()", request_id, details=str(last_err))

        predicted_label = pred_result.get_predicted_label()
        probabilities = to_jsonable(pred_result.get_probabilities())

        # --- 5) Save to DB ---
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
            else:
                # No matamos la request: devolvemos la predicción igual
                print(f"[{request_id}] ❌ No DB connection. DB_CONFIG={db_config_sanitized(DB_CONFIG)}")

        # Cleanup temp
        if TEST_IMAGE_PATH.exists():
            os.remove(TEST_IMAGE_PATH)

        return jsonify(
            {
                "prediccion": predicted_label,
                "meta": {"image_url": saved_url, "db_inserted": db_success},
                "probabilities": probabilities,
            }
        )

    except Exception as e:
        print(f"[{request_id}] ERROR in /predict: {e}")
        print(traceback.format_exc())
        return error_response(500, "Error processing image in backend", request_id, details=str(e))

    finally:
        if TEST_IMAGE_PATH.exists():
            os.remove(TEST_IMAGE_PATH)


@app.route("/predict/correcta", methods=["POST"])
def pred_correcta():
    request_id = uuid.uuid4().hex[:10]
    try:
        url = (request.form.get("url") or "").strip()
        if not url:
            return error_response(400, 'Falta "url".', request_id)

        db = DatabaseConnection(**DB_CONFIG)
        with db as conn:
            if not conn.cnx:
                return error_response(
                    500,
                    "No hay conexión a DB.",
                    request_id,
                    details={"db_config": db_config_sanitized(DB_CONFIG)},
                )

            ok = conn.update_label_pred_correcta(url)

        return jsonify({"success": bool(ok), "request_id": request_id}), (200 if ok else 500)

    except Exception as e:
        print(f"[{request_id}] ERROR en /predict/correcta: {e}")
        print(traceback.format_exc())
        return error_response(500, "Error procesando /predict/correcta", request_id, details=str(e))


@app.route("/predict/incorrecta", methods=["POST"])
def pred_incorrecta():
    request_id = uuid.uuid4().hex[:10]
    try:
        url = (request.form.get("url") or "").strip()
        label_raw = (request.form.get("label") or "").strip()

        if not url:
            return error_response(400, 'Falta "url".', request_id)
        if label_raw == "":
            return error_response(400, 'Falta "label".', request_id)

        try:
            label_int = int(label_raw)
        except ValueError:
            return error_response(400, '"label" debe ser un entero.', request_id, details={"got": label_raw})

        db = DatabaseConnection(**DB_CONFIG)
        with db as conn:
            if not conn.cnx:
                return error_response(
                    500,
                    "No hay conexión a DB.",
                    request_id,
                    details={"db_config": db_config_sanitized(DB_CONFIG)},
                )

            ok = conn.update_label_pred_incorrecta(url, label_int)

        return jsonify({"success": bool(ok), "request_id": request_id}), (200 if ok else 500)

    except Exception as e:
        print(f"[{request_id}] ERROR en /predict/incorrecta: {e}")
        print(traceback.format_exc())
        return error_response(500, "Error procesando /predict/incorrecta", request_id, details=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), debug=True)
