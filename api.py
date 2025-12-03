from flask import Flask, jsonify,request
from predict import prediction
import cv2

# 1. Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def hand_sing_predict():
    """
    Función que maneja las peticiones GET a /libros.
    Devuelve la lista de libros en formato JSON.
    """
    datos_json = request.get_json()
    
    if 'nombre' not in datos_json or 'imagen' not in datos_json:
        # Si faltan campos, retornamos un 400 con el mensaje de error
        return jsonify({'error': 'Parámetros obligatorios faltantes. Se requiere "nombre" y "imagen".'}), 400
    
    foto  = datos_json['imagen']
    nombre = datos_json['nombre']
    foto_path = f"datos/{nombre}"
    cv2.imwrite(foto_path, foto)
    
    prediccion = prediction(datos_json['nombre'])
    # en una respuesta JSON adecuada con las cabeceras correctas.
    return jsonify({'prediccion': prediccion})



if __name__ == '__main__':
    app.run(debug=True)