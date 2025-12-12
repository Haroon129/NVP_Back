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
# Importaciones para las mejoras
from tensorflow.keras.optimizers import AdamW 
from tensorflow.keras.regularizers import l2 

# ===========================
#   1. RUTAS DEL DATASET
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "src", "data", "dataset_1_12") 

# ===========================
#   2. DEFINICIÓN DEL MODELO 
# ===========================

def create_alphabet_classifier(input_shape=(256,256,1), num_classes=10):
    model = Sequential()
    
    # Bloque 1 (256 -> 128) - Añadido Regularización L2
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', 
                     input_shape=input_shape,
                     kernel_regularizer=l2(0.0001))) # <-- Regularización L2
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
    model.add(MaxPooling2D((2,2)))
    
    # Clasificador DENSE
    model.add(Flatten())
    model.add(Dense(1024, activation='relu')) 
    model.add(Dropout(0.2))                  
    model.add(Dense(num_classes, activation='softmax')) 
    return model

# ===========================
#   3. DATA AUGMENTATION Y CARGA DEL DATASET
# ===========================
img_size = (256, 256) 
batch_size = 32
seed_value = 42

# Generador para ENTRENAMIENTO (Robustez máxima)
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,  
    zoom_range=0.1, brightness_range=[0.7, 1.3], shear_range=0.1, horizontal_flip=True,       
    validation_split=0.2      
)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


print(f"Cargando y reescalando imágenes a {img_size} con Data Augmentation...")

# Generadores
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR, target_size=img_size, color_mode="grayscale", batch_size=batch_size,
    class_mode='sparse', subset='training', seed=seed_value
)
test_generator = test_datagen.flow_from_directory(
    DATASET_DIR, target_size=img_size, color_mode="grayscale", batch_size=batch_size,
    class_mode='sparse', subset='validation', seed=seed_value
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(f"Clases detectadas: {class_names}")

# ===========================
#   4. SELECCIÓN Y COMPILACIÓN DEL MODELO
# ===========================
model = create_alphabet_classifier(input_shape=(img_size[0], img_size[1], 1), 
                                   num_classes=num_classes)

# 💡 Usamos AdamW con un Learning Rate base
optimizer = AdamW(learning_rate=0.001) 
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
    factor=0.4,         # Reducción a 40% (más fino que 0.5)
    patience=3,         
    min_lr=0.00001
)

# ===========================
#   6. ENTRENAMIENTO
# ===========================
print("\nIniciando entrenamiento final (NOCTURNO: 256x256, AdamW, L2)...")
history = model.fit(
    train_generator, 
    epochs=200, # Aumentado para aprovechar el tiempo
    validation_data=test_generator, 
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)

# ===========================
#   7. GUARDAR MODELO
# ===========================
MODEL_PATH = os.path.join(BASE_DIR, "models","model_Digits_2.keras") 
model.save(MODEL_PATH)
print(f"\nModelo entrenado (V4) y guardado en: {MODEL_PATH}")