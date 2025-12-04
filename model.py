import os
# --- OPTIMIZACIÓN: Silenciar advertencias de TensorFlow ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = Mostrar solo errores
# --------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Importado para Augmentation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===========================
#   1. RUTAS DEL DATASET
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta a la carpeta que contiene las subcarpetas de clase '0' a '9'
DATASET_DIR = os.path.join(BASE_DIR, "src", "data", "dataset_1_12") 

# ===========================
#   2. DEFINICIÓN DEL MODELO (CNN para 128x128)
# ===========================

def create_alphabet_classifier(input_shape=(128,128,1), num_classes=10):
    """Crea un modelo CNN adaptado para imágenes de 128x128 en escala de grises."""
    model = Sequential()
    
    # Bloque 1 (128 -> 64)
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Bloque 2 (64 -> 32)
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Bloque 3 (32 -> 16)
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    
    # Bloque 4 (16 -> 8)
    model.add(Conv2D(256, (3,3), activation='relu', padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Bloque 5 (8 -> 4)
    model.add(Conv2D(256, (3,3), activation='relu', padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    
    # Clasificador DENSE
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) 
    model.add(Dropout(0.3))                  
    model.add(Dense(num_classes, activation='softmax')) 
    return model

# ===========================
#   3. DATA AUGMENTATION Y CARGA DEL DATASET
# ===========================
img_size = (128, 128)
batch_size = 32
seed_value = 42

# 🚨 Generador para ENTRENAMIENTO: Aplica Aumento de Datos para robustez 
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalización
    rotation_range=15,        # Aumento: Rotación
    width_shift_range=0.15,   # Aumento: Desplazamiento
    height_shift_range=0.15,  # Aumento: Desplazamiento
    zoom_range=0.1,           # Aumento: Zoom
    brightness_range=[0.8, 1.2], # Aumento: Simula cambios de iluminación 💡
    shear_range=0.1,          # Aumento: Sesgado
    validation_split=0.2      # 20% para validación
)

# Generador para VALIDACIÓN: Solo normalización
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


print(f"Cargando y reescalando imágenes a {img_size} con Data Augmentation...")

# Generador de Entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=img_size,
    color_mode="grayscale", # 1 canal
    batch_size=batch_size,
    class_mode='sparse',    # Etiquetas como enteros (0, 1, 2...)
    subset='training',
    seed=seed_value
)

# Generador de Validación/Prueba
test_generator = test_datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    seed=seed_value
)


class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(f"Clases detectadas: {class_names}")
print(f"Número de clases: {num_classes}")

# ===========================
#   4. SELECCIÓN Y COMPILACIÓN DEL MODELO
# ===========================
model = create_alphabet_classifier(input_shape=(img_size[0], img_size[1], 1), 
                                   num_classes=num_classes)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ===========================
#   5. CALLBACKS
# ===========================
# La paciencia de EarlyStopping podría necesitar ser más alta (ej. 7 o 10) 
# debido a que el Data Augmentation introduce más ruido.
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=7,         # Aumentado de 5 a 7 para dar más tiempo a la convergencia
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,         
    patience=3,         # Aumentado de 2 a 3 para dar más tiempo al nuevo LR
    min_lr=0.00001
)

# ===========================
#   6. ENTRENAMIENTO
# ===========================
print("\nIniciando entrenamiento con ROBUSTEZ MEJORADA...")
history = model.fit(
    train_generator, 
    epochs=50, 
    validation_data=test_generator, 
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)

# ===========================
#   7. GUARDAR MODELO
# ===========================
MODEL_PATH = os.path.join(BASE_DIR, "src","models","model_Digits_ROBUST.h5") # Nuevo nombre
model.save(MODEL_PATH)
print(f"\nModelo entrenado (ROBUSTO) y guardado en: {MODEL_PATH}")