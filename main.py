# ResNet Training on CIFAR-10 Dataset - Google Colab Implementation

# ===== INSTALACIÓN Y CONFIGURACIÓN =====
# Ejecutar primero en Colab para verificar GPU
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU disponible:", tf.test.is_gpu_available())
print("Dispositivos:", tf.config.list_physical_devices())

# ===== IMPORTACIÓN DE LIBRERÍAS =====
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ===== CONFIGURACIÓN DE PARÁMETROS =====
# Configuración optimizada - usando tamaño original de CIFAR-10
BATCH_SIZE = 64  # Podemos usar batch más grande sin redimensionar
EPOCHS = 100
IMG_HEIGHT = 32  # Tamaño original de CIFAR-10
IMG_WIDTH = 32   # Tamaño original de CIFAR-10
NUM_CLASSES = 10
VALIDATION_SPLIT = 0.2

# Configurar límites de memoria GPU
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Configuración de memoria GPU establecida")
except RuntimeError as e:
    print("Error configurando GPU:", e)

# Nombres de las clases CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# ===== CARGA Y EXPLORACIÓN DE DATOS =====
print("Cargando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Datos de entrenamiento: {x_train.shape}")
print(f"Labels de entrenamiento: {y_train.shape}")
print(f"Datos de prueba: {x_test.shape}")
print(f"Labels de prueba: {y_test.shape}")

# Visualizar algunas imágenes del dataset
plt.figure(figsize=(12, 8))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(f'{class_names[y_train[i][0]]}')
    plt.axis('off')
plt.suptitle('Ejemplos del Dataset CIFAR-10')
plt.tight_layout()
plt.show()

# ===== PREPROCESAMIENTO DE DATOS =====
print("Preprocesando datos...")

# Normalización (solo esto es necesario)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding para las labels
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

print(f"Forma después del preprocesamiento:")
print(f"X_train: {x_train.shape}")  # (50000, 32, 32, 3)
print(f"Y_train: {y_train_cat.shape}")
print("¡No redimensionamiento necesario! Usamos el tamaño original 32x32")

# ===== DATA AUGMENTATION =====
print("Configurando data augmentation...")

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# No augmentation para datos de validación
val_datagen = ImageDataGenerator(validation_split=VALIDATION_SPLIT)

# Generadores de datos (usando las imágenes originales)
train_generator = train_datagen.flow(
    x_train, y_train_cat,  # Usar x_train original, no redimensionado
    batch_size=BATCH_SIZE,
    subset='training'
)

validation_generator = val_datagen.flow(
    x_train, y_train_cat,  # Usar x_train original, no redimensionado
    batch_size=BATCH_SIZE,
    subset='validation'
)

# ===== CONSTRUCCIÓN DEL MODELO RESNET =====
print("Construyendo modelo ResNet...")

# Cargar ResNet50 pre-entrenado (sin la capa superior)
# IMPORTANTE: Cambiar input_shape para aceptar imágenes de 32x32
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)  # Usar el tamaño original de CIFAR-10
)

# Congelar las primeras capas para transfer learning
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Agregar capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Crear el modelo completo
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Mostrar resumen del modelo
print("Resumen del modelo:")
model.summary()

# ===== CALLBACKS =====
print("Configurando callbacks...")

# Checkpoint para guardar el mejor modelo
checkpoint = ModelCheckpoint(
    'best_resnet_cifar10.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Early stopping para evitar overfitting
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Reducir learning rate cuando el loss se estanque
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ===== ENTRENAMIENTO =====
print("Iniciando entrenamiento...")

# Calcular steps por epoch
steps_per_epoch = int(len(x_train) * (1 - VALIDATION_SPLIT) // BATCH_SIZE)
validation_steps = int(len(x_train) * VALIDATION_SPLIT // BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# ===== EVALUACIÓN =====
print("Evaluando modelo en datos de prueba...")

# Preparar datos de prueba (usando el tamaño original)
test_predictions = model.predict(x_test, batch_size=BATCH_SIZE)  # x_test original
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = np.argmax(y_test_cat, axis=1)

# Calcular accuracy final
test_accuracy = np.mean(test_pred_classes == test_true_classes)
print(f"Accuracy en datos de prueba: {test_accuracy:.4f}")

# ===== VISUALIZACIÓN DE RESULTADOS =====
print("Generando visualizaciones...")

# Gráficas de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Matriz de confusión
plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_true_classes, test_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Verdad')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(test_true_classes, test_pred_classes, 
                          target_names=class_names))

# ===== PREDICCIONES DE EJEMPLO =====
print("Mostrando predicciones de ejemplo...")

# Seleccionar algunas imágenes de prueba aleatoriamente
indices = np.random.choice(len(x_test), 12, replace=False)

plt.figure(figsize=(15, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 4, i + 1)
    
    # Mostrar imagen original (32x32)
    plt.imshow(x_test[idx])
    
    # Hacer predicción (usando imagen original)
    pred = model.predict(x_test[idx:idx+1], verbose=0)  # x_test original
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    true_class = y_test[idx][0]
    
    # Color: verde si correcto, rojo si incorrecto
    color = 'green' if pred_class == true_class else 'red'
    
    plt.title(f'Pred: {class_names[pred_class]}\n'
              f'True: {class_names[true_class]}\n'
              f'Conf: {confidence:.2f}', 
              color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('Predicciones del Modelo ResNet50 en CIFAR-10')
plt.tight_layout()
plt.show()

# ===== GUARDAR MODELO FINAL =====
print("Guardando modelo final...")
model.save('resnet50_cifar10_final.h5')

# Análisis de accuracy por clase
print("\nAccuracy por clase:")
for i, class_name in enumerate(class_names):
    class_mask = (test_true_classes == i)
    class_accuracy = np.mean(test_pred_classes[class_mask] == test_true_classes[class_mask])
    print(f"{class_name}: {class_accuracy:.4f}")

print("\n¡Entrenamiento completado exitosamente!")
print(f"Modelo guardado como 'resnet50_cifar10_final.h5'")
print(f"Mejor modelo guardado como 'best_resnet_cifar10.h5'")