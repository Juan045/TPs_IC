
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.data import AUTOTUNE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. CONFIGURACIÓN Y CARGA DEL DATASET

dataset_dir = os.path.normpath("D:/ASL_Alphabet_Dataset/asl_alphabet_train")

img_height, img_width = 224, 224
batch_size = 16
val_split = 0.2
seed = 123

# Cargar datasets (sin pad_to_aspect_ratio)
train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=val_split,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=val_split,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Guardar nombres de clases
class_names = train_ds.class_names
num_classes = len(class_names)
print("Número de clases:", num_classes)
print("Clases:", class_names)

# Preprocesamiento: mantener aspecto y normalizar
def preprocess(image, label):
    image = tf.image.resize_with_pad(image, img_height, img_width)
    image = preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

# Cache y prefetch
train_ds = train_ds.shuffle(1000, reshuffle_each_iteration=True)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. CONSTRUCCIÓN DEL MODELO RESNET50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False  # congelar capas convolucionales

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 🚀 3. ENTRENAMIENTO (FASE 1: capas congeladas)
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model_resnet50.h5', save_best_only=True, monitor='val_accuracy')
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# 🔧 4. FINE-TUNING (descongelar últimas 50 capas)

for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

# 5. EVALUACIÓN Y MÉTRICAS
loss, acc = model.evaluate(val_ds)
print(f"\n✅ Accuracy final: {acc*100:.2f}%")
print(f"📉 Pérdida final: {loss:.4f}")

# Predicciones
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(12, 12))
disp.plot(xticks_rotation=90, cmap='Blues', colorbar=False)
plt.title("Matriz de Confusión - ResNet50")
plt.show()

# 6. GRÁFICOS DE ENTRENAMIENTO

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento (fase 1)')
plt.plot(history.history['val_accuracy'], label='Validación (fase 1)')
plt.plot(history_ft.history['accuracy'], label='Entrenamiento (fine-tuning)')
plt.plot(history_ft.history['val_accuracy'], label='Validación (fine-tuning)')
plt.title('Evolución de la Exactitud (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento (fase 1)')
plt.plot(history.history['val_loss'], label='Validación (fase 1)')
plt.plot(history_ft.history['loss'], label='Entrenamiento (fine-tuning)')
plt.plot(history_ft.history['val_loss'], label='Validación (fine-tuning)')
plt.title('Evolución de la Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.show()

print("\n✅ Entrenamiento y evaluación finalizados correctamente.")
