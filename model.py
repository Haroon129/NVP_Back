import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ===========================
#   RUTAS DEL DATASET
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "src", "data","dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
TEST_DIR = os.path.join(DATASET_DIR, "Test")

# ===========================
#    DEFINICIÓN DEL MODELO
# ===========================

def create_binary_classifier(input_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))  # salida binaria
    return model


def create_alphabet_classifier(input_shape=(28,28,1), num_classes=24):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))  # salida multiclase
    return model

# ===========================
#   DATA AUGMENTATION
# ===========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ===========================
#   CARGA DEL DATASET
# ===========================
img_size = (28,28)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=img_size,
    color_mode="grayscale",
    batch_size=32
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=img_size,
    color_mode="grayscale",
    batch_size=32
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Clases detectadas:", class_names)

# ===========================
#   SELECCIÓN DEL MODELO
# ===========================
modo = input("Selecciona modelo:\n   1 - Binario A vs No-A \n   2 - Multiclase A-Z\n")
if modo == '1':
    model = create_binary_classifier()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    model = create_alphabet_classifier(num_classes=num_classes)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# ===========================
#   ENTRENAMIENTO
# ===========================
if modo == '1':
    # Convertir labels a binario A / no-A
    A_index = class_names.index('A')
    def to_binary(ds):
        return ds.map(lambda x, y: (x/255.0, tf.expand_dims(tf.cast(tf.equal(y, A_index), tf.float32), -1)))
    train_bin = to_binary(train_ds)
    test_bin = to_binary(test_ds)
    history = model.fit(train_bin, 
                        epochs=20, 
                        validation_data=test_bin, 
                        verbose=2)
else:
    history = model.fit(train_ds.map(lambda x, y: (x/255.0, y)), 
                        epochs=20, 
                        validation_data=test_ds.map(lambda x, y: (x/255.0, y)), 
                        verbose=2)

# ===========================
#   GUARDAR MODELO
# ===========================
MODEL_PATH = os.path.join(BASE_DIR, "src","models","model_prueba.h5")
model.save(MODEL_PATH)
print(f"\nModelo entrenado y guardado en: {MODEL_PATH}")
