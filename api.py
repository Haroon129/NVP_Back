from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
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

ROOT = Path(__file__).resolve().parent

# Directories and configuration
PREDICT_DIR = ROOT / "data" / "predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the temporary prediction directory exists
TEST_IMAGE_PATH = PREDICT_DIR / "test.jpg"


# Load environment variables from the .env file
load_dotenv() 

# Database Configuration (using os.environ)
DB_CONFIG = {
    'host': os.environ.get("DB_HOST"), 
    'user': os.environ.get("DB_USER"), 
    'database': os.environ.get("DB_DATABASE"), 
    'password': os.environ.get("DB_PASSWORD", ""), # Use "" as default if not found
}

# =========================================================
# AUXILIARY FUNCTIONS
# =========================================================

def to_jsonable(obj):
    """
    Recursively converts non-JSON serializable objects (Path, bytes, numpy types) 
    into JSON-friendly types (str, int, float, list, dict).

    Args:
        obj: The object to be converted.

    Returns:
        The JSON-serializable representation of the object.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (bytes, bytearray)):
        # Encode bytes to base64 string
        return base64.b64encode(obj).decode("utf-8")

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        # Ensure dictionary keys are strings
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
        # Fallback if numpy is not available or type is unsupported
        pass
    
    return str(obj)


def error_response(status_code: int, error: str, request_id: str, details=None, extra=None):
    """
    Constructs a standardized JSON error response.

    Args:
        status_code (int): The HTTP status code for the response.
        error (str): A brief, user-readable description of the error.
        request_id (str): The unique ID generated for the request (for tracing).
        details (Any, optional): Detailed technical information about the error. Defaults to None.
        extra (Any, optional): Additional metadata. Defaults to None.

    Returns:
        tuple: A Flask response (jsonified payload, status_code).
    """
    payload = {
        "error": error,
        "request_id": request_id,
    }
    if details is not None:
        payload["details"] = to_jsonable(details) # Use to_jsonable for safe output
    if extra is not None:
        payload["extra"] = to_jsonable(extra)
    return jsonify(payload), status_code

# =========================================================
# API ENDPOINTS
# =========================================================

@app.route("/predict", methods=["POST"])
def hand_sign_predict():
    """
    Primary API endpoint for hand sign prediction. 
    
    Handles image upload via:
    1. multipart/form-data (File upload)
    2. application/json (Base64 string)
    
    The process includes saving the image, uploading to Cloudinary, 
    running the ML prediction, and logging the result to the database.

    Returns:
        tuple: A Flask response (jsonified prediction result, HTTP status code).
    """
    request_id = uuid.uuid4().hex[:10]
    
    saved_url = None
    real_label = -1
    img_bytes = None
    
    try:
        nombre = None
        unique_name = None

        # --- 1. HANDLE INPUT (File or Base64) ---
        if request.files and "imagen" in request.files:
            # --- File Upload (multipart/form-data) ---
            file = request.files.get("imagen")
            nombre = (request.form.get("nombre") or "").strip()
            
            try:
                # Optionally get the ground truth label from the form data
                real_label = int(request.form.get("label", -1))
            except (ValueError, TypeError):
                pass
            
            if not nombre:
                return error_response(400, 'Missing field "nombre".', request_id)
            if file is None or not file.filename:
                return error_response(400, 'Missing file "imagen".', request_id)

            original_filename = secure_filename(file.filename)
            ext = Path(original_filename).suffix.lower() or ".jpg"
            # Create a unique name using the provided 'nombre' and a UUID
            unique_name = f"{secure_filename(nombre)}_{uuid.uuid4().hex}{ext}"
            
            file.seek(0)
            img_bytes = file.read()

        else:
            # --- Base64 JSON Payload (application/json) ---
            data = request.get_json(silent=True) or {}
            nombre = (data.get("nombre") or "").strip()
            foto_b64 = (data.get("imagen") or "").strip()
            
            try:
                # Optionally get the ground truth label from the JSON data
                real_label = int(data.get("label", -1))
            except (ValueError, TypeError):
                pass

            if not nombre or not foto_b64:
                return error_response(
                    400, 'Missing "nombre" and/or "imagen".', request_id, 
                    details={"got_keys": list(data.keys())}
                )

            # Clean up potential data URI prefix (e.g., data:image/jpeg;base64,)
            foto_b64 = re.sub(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", "", foto_b64)
            
            try:
                img_bytes = base64.b64decode(foto_b64, validate=True)
            except Exception as e:
                return error_response(400, "Invalid Base64 in 'imagen'.", request_id, details=str(e))
            
            unique_name = f"{secure_filename(nombre)}_{uuid.uuid4().hex}.jpg"
            
        # 2. Save Image Temporarily for Prediction
        if img_bytes is None:
             return error_response(500, "Could not retrieve image content.", request_id)
             
        TEST_IMAGE_PATH.write_bytes(img_bytes)

        # 3. Upload to Cloudinary
        public_id_name = Path(unique_name).stem 
        saved_url = subir_imagen_a_cloudinary(public_id_name, img_bytes)
        
        if saved_url is None:
            return error_response(
                500,
                "Error uploading image to Cloudinary.",
                request_id
            )

        # 4. EXECUTE PREDICTION
        pred_result = None
        last_err = None
        
        # Try different naming conventions for prediction (robustness)
        for arg in (str(TEST_IMAGE_PATH), TEST_IMAGE_PATH.name, "test.jpg", nombre):
            try:
                # Assuming prediction() takes the file path or a name and returns a result object
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
        
        # 5. SAVE TO DATABASE (NVP.imagenes)
        db_conn = DatabaseConnection(**DB_CONFIG)
        
        insert_sql = """
        INSERT INTO imagenes (URL, predicted_label, label) 
        VALUES (%s, %s, %s)
        """
        # Insert the Cloudinary URL, the ML prediction, and the optional ground truth label
        insert_params = (saved_url, predicted_label, real_label)
        
        db_success = False
        # Use the Context Manager for safe database connection handling
        with db_conn as db:
            if db.cnx:
                db_success = db.execute_query(insert_sql, insert_params)
        
        # CLEANUP: Remove the temporary file
        if TEST_IMAGE_PATH.exists():
            os.remove(TEST_IMAGE_PATH)
        
        meta = {
            "image_url": saved_url,
            "db_inserted": db_success
        }
        
        # Final success response
        return jsonify({
            "prediccion": predicted_label, 
            "meta": meta, 
            "probabilities": to_jsonable(pred_result.get_probabilities())
        })

    except Exception as e:
        # Generic error handler for unexpected exceptions
        print(f"[{request_id}] ERROR in /predict: {e}")
        print(traceback.format_exc())
        return error_response(500, "Error processing image in backend", request_id, details=str(e))
        
    finally:
        # Double-check cleanup (though already handled within the try block)
        if TEST_IMAGE_PATH.exists():
            os.remove(TEST_IMAGE_PATH)


@app.route('/predict/correcta', methods=["POST"])
def pred_correcta():
    """
    Endpoint to correct a prediction (Mark prediction as correct).
    
    Expects a 'url' in the form data. Assumes that DatabaseConnection 
    has a method 'update_label_pred_correcta' to handle the DB update logic.

    Returns:
        bool: True if the database update was successful, False otherwise.
    """
    url = request.form.get('url')
    # NOTE: Assumes update_label_pred_correcta is implemented in DatabaseConnection
    db = DatabaseConnection(**DB_CONFIG) 
    # Must use Context Manager or close connection explicitly if not using 'with'
    with db as conn:
        if conn.cnx and hasattr(conn, 'update_label_pred_correcta') and conn.update_label_pred_correcta(url):
            return jsonify(success=True), 200
    return jsonify(success=False, error="DB operation failed"), 500


@app.route('/predict/incorrecta',methods=["POST"])
def pred_incorrecta():
    """
    Endpoint to correct a prediction (Mark prediction as incorrect and update the label).

    Expects 'url' and the 'label' in the form data. Assumes that DatabaseConnection 
    has a method 'update_label_pred_incorrecta' to handle the DB update logic.

    Returns:
        bool: True if the database update was successful, False otherwise.
    """
    url = request.form.get('url')
    label = request.form.get('label')
    # NOTE: Assumes update_label_pred_incorrecta is implemented in DatabaseConnection
    db = DatabaseConnection(**DB_CONFIG)
    with db as conn:
        if conn.cnx and hasattr(conn, 'update_label_pred_incorrecta') and conn.update_label_pred_incorrecta(url, label):
            return jsonify(success=True), 200
    return jsonify(success=False, error="DB operation failed"), 500


if __name__ == "__main__":
    # Start the Flask application
    app.run(
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", "5001")), 
        debug=True # debug=True is often discouraged in production
    )