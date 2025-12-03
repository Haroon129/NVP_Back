from flask import Flask, jsonify, request
from predict import prediction
import cv2
import numpy as np
import os

app = Flask(__name__)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

@app.route("/predict", methods=["POST"])
def hand_sing_predict():
    # multipart/form-data esperado:
    # - nombre: string (opcional, pero recomendado)
    # - imagen: file (obligatorio)
    nombre = request.form.get("nombre")
    file = request.files.get("imagen")

    if file is None:
        return jsonify(
            {"error": 'Parámetros obligatorios faltantes. Se requiere "imagen" (file) y "nombre" (string opcional).'}
        ), 400

    # Si no llega nombre, usamos el filename del upload
    if not nombre:
        nombre = file.filename or "captura.jpg"

    # Asegurar carpeta y extensión
    os.makedirs("datos", exist_ok=True)

    base, ext = os.path.splitext(nombre)
    if ext.lower() not in ALLOWED_EXTS:
        nombre = f"{nombre}.jpg"  # fuerza extensión válida

    foto_path = os.path.join("datos", nombre)

    try:
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"error": "El archivo recibido está vacío."}), 400

        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "No se pudo decodificar la imagen (archivo inválido)."}), 400

        ok = cv2.imwrite(foto_path, img)
        if not ok:
            return jsonify({"error": "No se pudo guardar la imagen con cv2.imwrite", "path": foto_path}), 500

        # Si tu prediction espera ruta, usa prediction(foto_path)
        prediccion = prediction(nombre)

        return jsonify(
            {
                "prediccion": prediccion,
                "meta": {
                    "saved_as": nombre,
                    "path": foto_path,
                    "content_type": file.content_type,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": "Error procesando imagen en backend", "details": str(e)}), 500


if __name__ == "__main__":
    # Ajusta el puerto si tu servidor corre en 5001, por ejemplo.
    app.run(host="0.0.0.0", port=5001, debug=True)
