from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import base64
import re
import traceback
import os

from predict import prediction

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent

PREDICT_DIR = ROOT / "data" / "predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGE_PATH = PREDICT_DIR / "test.jpg"


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
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            return to_jsonable(dataclasses.asdict(obj))
    except Exception:
        pass

    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return to_jsonable(obj.dict())
        except Exception:
            pass

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return to_jsonable(obj.to_dict())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            # ojo: __dict__ puede contener cosas no serializables; por eso recursivo
            return to_jsonable(vars(obj))
        except Exception:
            pass

    return str(obj)


def error_response(status_code: int, error: str, request_id: str, details=None, extra=None):
    payload = {
        "error": error,
        "request_id": request_id,
    }
    if details is not None:
        payload["details"] = details
    if extra is not None:
        payload["extra"] = extra
    return jsonify(payload), status_code


@app.route("/predict", methods=["POST"])
def hand_sign_predict():
    request_id = uuid.uuid4().hex[:10]

    try:
        saved_path = None
        nombre = None

        if request.files and "imagen" in request.files:
            file = request.files.get("imagen")
            nombre = (request.form.get("nombre") or "").strip()

            if not nombre:
                return error_response(400, 'Falta el campo "nombre" (form-data).', request_id)
            if file is None or not file.filename:
                return error_response(400, 'Falta el archivo "imagen" (form-data).', request_id)

            original_filename = secure_filename(file.filename)
            ext = Path(original_filename).suffix.lower() or ".jpg"
            unique_name = f"{secure_filename(nombre)}_{uuid.uuid4().hex}{ext}"
            unique_path = PREDICT_DIR / unique_name
            file.save(unique_path)

            TEST_IMAGE_PATH.write_bytes(unique_path.read_bytes())
            saved_path = TEST_IMAGE_PATH

        else:
            data = request.get_json(silent=True) or {}
            nombre = (data.get("nombre") or "").strip()
            foto_b64 = (data.get("imagen") or "").strip()

            if not nombre or not foto_b64:
                return error_response(
                    400,
                    'Faltan "nombre" y/o "imagen". Se requiere JSON con ambos campos.',
                    request_id,
                    details={"got_keys": list(data.keys())},
                )

            foto_b64 = re.sub(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", "", foto_b64)

            try:
                img_bytes = base64.b64decode(foto_b64, validate=True)
            except Exception as e:
                return error_response(400, "Base64 inválido en 'imagen'.", request_id, details=str(e))

            TEST_IMAGE_PATH.write_bytes(img_bytes)
            saved_path = TEST_IMAGE_PATH

        if saved_path is None or not saved_path.exists():
            return error_response(
                500,
                "No se pudo guardar la imagen correctamente.",
                request_id,
                details={"expected": str(TEST_IMAGE_PATH)},
            )

        pred = None
        last_err = None

        for arg in (str(saved_path), saved_path.name, "test.jpg", nombre):
            try:
                pred = prediction(arg)
                last_err = None
                break
            except Exception as e:
                last_err = e

        if pred is None:
            print(f"[{request_id}] prediction() falló. Last error: {last_err}")
            print(traceback.format_exc())
            return error_response(
                500,
                "Error ejecutando prediction()",
                request_id,
                details=str(last_err),
                extra={"saved_path": str(saved_path), "tried_args": [str(saved_path), saved_path.name, "test.jpg", nombre]},
            )

        pred_json = to_jsonable(pred)

        meta = {
            "saved_path": str(saved_path),
            "prediction_type": type(pred).__name__,
        }

        return jsonify({"prediccion": pred_json, "meta": meta, "request_id": request_id})

    except Exception as e:
        print(f"[{request_id}] ERROR en /predict: {e}")
        print(traceback.format_exc())
        return error_response(
            500,
            "Error procesando imagen en backend",
            request_id,
            details=str(e),
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), debug=True)
