# NVP_Back — Backend (Flask + TensorFlow + OpenCV)

Backend en Flask que expone un endpoint `POST /predict` para recibir una imagen (multipart/form-data) y devolver una predicción del modelo.

- **Puerto por defecto:** `5001`
- **Endpoint:** `POST http://localhost:5001/predict`
- **Body esperado (form-data):**
  - `nombre` (text) — nombre único del archivo (recomendado incluir extensión, ej: `hand_123.jpg`)
  - `imagen` (file) — archivo de imagen

---

## Requisitos

- Docker + Docker Compose instalados

---

## Ejecutar con Docker (recomendado)

### 1) Levantar en segundo plano
```bash
docker compose up -d --build
