import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# dataset
from tensorflow.keras.datasets import cifar10
# model
from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input, Resizing
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TensorFlow version:", tf.__version__)
print("GPU disponible:", tf.test.is_gpu_available())
print("Dispositivos:", tf.config.list_physical_devices())

# Configurar memoria GPU
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Configuración de memoria GPU establecida")
except RuntimeError as e:
    print("Error configurando GPU:", e)

# ===== PARÁMETROS =====
BATCH_SIZE = 128
EPOCHS = 80
NUM_CLASSES = 10
VALIDATION_SPLIT = 0.2
HEIGHT = 32
WIDTH = 32

# El 80% de los datos se usará para entrenamiento y el 20% para validación
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from tensorflow.keras.applications.resnet50 import preprocess_input
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

print(x_train.shape)

# One-hot encoding
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)


# Crear modelo
model = Sequential()

resnet_model = ResNet50(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg",
    classes=NUM_CLASSES,
    weights="imagenet"
)

for layer in resnet_model.layers:
    layer.trainable = False


model.add(Resizing(224, 224, input_shape=(32, 32, 3)))
model.add(resnet_model)
#model.add(GlobalAveragePooling2D)
#model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation="softmax")) # Ultima capa de clasificacion

model.summary()

# ===== Compilar el modelo =====
# Probamos el optimizador Adam
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ===== CALLBACKS =====
checkpoint = ModelCheckpoint('best_resnet_cifar10.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
callbacks = [checkpoint, early_stopping]

# ===== Entrenamiento =====
history = model.fit(
    x=x_train,
    y=y_train_cat,
    validation_data=(x_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ===== EVALUACIÓN =====
test_predictions = model.predict(x_test, batch_size=BATCH_SIZE)
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = np.argmax(y_test_cat, axis=1)
test_accuracy = np.mean(test_pred_classes == test_true_classes)
print(f"Accuracy en test: {test_accuracy:.4f}")

# ===== VISUALIZACIÓN =====
# Accuracy y loss
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(True)
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Loss')
ax2.legend()
ax2.grid(True)
plt.show()

# Matriz de confusión
cm = confusion_matrix(test_true_classes, test_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Verdad')
plt.show()

# Reporte de clasificación
print(classification_report(test_true_classes, test_pred_classes, target_names=class_names))