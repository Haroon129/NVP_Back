# 💻 NVP_Back — None Verbal People (Backend)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md) 
## 📝 Descripción del Proyecto

Este proyecto backend, denominado **NVP**, aloja una Red Neuronal Recurrente (RNN) implementada con **TensorFlow** para la **Predicción de lenguaje de señas**.

El backend se construye con **Flask** y expone una API para la inferencia, gestionando:
* El entrenamiento del modelo a través de `model.py`.
* La realización de predicciones en tiempo real mediante el script `predict.py` y el servicio API.
* El manejo de datos de imagen con **OpenCV** y la clase de datos `fotografia.py`.

## ⚙️ Tecnologías Principales

| Categoría | Tecnología |
| :--- | :--- |
| **Lenguaje** | Python (Versión especificada en `.python-version`) |
| **Framework API** | Flask |
| **Framework ML** | TensorFlow (RNN) |
| **Librerías de Visión** | OpenCV |
| **Gestor de Dependencias** | `uv` |
| **Almacenamiento** | Cloudinary |

## 🚀 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

1.  **Python:** La versión especificada en el archivo `.python-version`.
2.  **uv:** El gestor de paquetes `uv`.

## 📦 Instalación y Configuración

Sigue estos pasos para configurar el proyecto localmente.

1. uv sync --Sincroniza las depentencias del proyecto y crea el entorno virtual
2. uv run api.py --Ejecuta la api en local

### 1. Clonar el repositorio

```bash
git clone [https://aws.amazon.com/es/what-is/repo/](https://aws.amazon.com/es/what-is/repo/)  # URL de ejemplo
cd NVP_BACK