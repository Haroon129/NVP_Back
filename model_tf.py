import os
# --- OPTIMIZACIÓN: Silenciar advertencias de TensorFlow ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = Mostrar solo errores
# --------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===========================
#   1. RUTAS DEL DATASET
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "src", "data", "dataset_1_12") 

# ===========================
#   2. DEFINICIÓN DEL MODELO BINARIO
# ===========================

def create_binary_classifier(input_shape=(256,256,1)):
    """
    CNN con la misma arquitectura V4, pero configurada para salida binaria (0 vs No-0).
    """
    model = Sequential()
    
    # Bloque 1 (256 -> 128)
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Bloque 2 (128 -> 64)
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Bloque 3 (64 -> 32)
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    
    # Bloque 4 (32 -> 16)
    model.add(Conv2D(256, (3,3), activation='relu', padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Bloque 5 (16 -> 8)
    model.add(Conv2D(256, (3,3), activation='relu', padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2))) # La salida espacial es 8x8
    
    # Clasificador DENSE
    model.add(Flatten())
    model.add(Dense(1024, activation='relu')) 
    model.add(Dropout(0.2))                  
    
    # 🚨 CAMBIO CLAVE: Salida binaria (1 neurona, activación Sigmoid)
    model.add(Dense(1, activation='sigmoid')) 
    return model

# ===========================
#   3. DATA AUGMENTATION Y CARGA DEL DATASET
# ===========================
img_size = (256, 256) 
batch_size = 32
seed_value = 42

# 🚨 CAMBIO CLAVE: Usamos class_mode='binary' para la salida Sigmoid
# Esto fuerza a que las etiquetas sean 0s y 1s.
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,  
    zoom_range=0.1, brightness_range=[0.7, 1.3], shear_range=0.1, horizontal_flip=True,       
    validation_split=0.2      
)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

print(f"Cargando y reescalando imágenes a {img_size} con Data Augmentation...")

# Generador de Entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='binary',    # 🚨 Usamos binary
    subset='training',
    seed=seed_value
)

# Generador de Validación/Prueba
test_generator = test_datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='binary',    # 🚨 Usamos binary
    subset='validation',
    seed=seed_value
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
# Notarás que el generador ordena las clases. '0' debe ser el índice 0.
print(f"Clases detectadas: {class_names}")

# ===========================
#   4. SELECCIÓN Y COMPILACIÓN DEL MODELO
# ===========================
model = create_binary_classifier(input_shape=(img_size[0], img_size[1], 1))

# 🚨 CAMBIO CLAVE: Usamos binary_crossentropy y métrica 'binary_accuracy'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# model.summary()

# ===========================
#   5. CALLBACKS
# ===========================
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=7,         
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,         
    patience=3,         
    min_lr=0.00001
)

# ===========================
#   6. ENTRENAMIENTO
# ===========================
print("\nIniciando entrenamiento BINARIO (0 vs NO-0)...")
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
MODEL_PATH = os.path.join(BASE_DIR, "src","models","model_Digits_Binary_0_vs_Not0.keras") 
model.save(MODEL_PATH)
print(f"\nModelo BINARIO entrenado y guardado en: {MODEL_PATH}")