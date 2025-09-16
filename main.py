import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from gpu import setup_gpu_training

# 1. Configurar GPU (si está disponible)
gpu_config = setup_gpu_training()

# 2. Cargar y preprocesar CIFAR-10
print("Cargando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Nombres de las clases
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Datos de entrenamiento: {x_train.shape}")
print(f"Datos de prueba: {x_test.shape}")
print(f"Clases: {len(class_names)}")

# 3. Preprocesamiento específico para ResNet
def preprocess_data(x_train, x_test, y_train, y_test):
    """Preprocesa los datos para ResNet50"""
    
    # Normalizar píxeles al rango [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # ResNet50 espera imágenes de al menos 32x32, CIFAR-10 ya es 32x32
    # Pero aplicamos el preprocesamiento estándar de ImageNet
    x_train = tf.keras.applications.resnet50.preprocess_input(x_train * 255.0)
    x_test = tf.keras.applications.resnet50.preprocess_input(x_test * 255.0)
    
    # Convertir etiquetas a categorical
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, x_test, y_train, y_test, num_classes

x_train, x_test, y_train, y_test, num_classes = preprocess_data(x_train, x_test, y_train, y_test)
print("Datos preprocesados correctamente!")

# 4. Crear modelo con ResNet50 pre-entrenada
def create_resnet50_model(num_classes=10, input_shape=(32, 32, 3)):
    """
    Crea modelo usando ResNet50 pre-entrenada con transfer learning
    """
    
    # Cargar ResNet50 pre-entrenada (sin las capas superiores)
    base_model = keras.applications.ResNet50(
        weights='imagenet',      # Pesos pre-entrenados en ImageNet
        include_top=False,       # Excluir la capa de clasificación final
        input_shape=input_shape
    )
    
    # Congelar las capas base para transfer learning
    base_model.trainable = False
    
    # Añadir capas personalizadas de clasificación
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='ResNet50_CIFAR10')
    
    return model, base_model

print("\nCreando modelo ResNet50...")
model, base_model = create_resnet50_model()

# Mostrar resumen del modelo
print("\nResumen del modelo:")
model.summary()

print(f"\nParámetros totales: {model.count_params():,}")
print(f"Parámetros entrenables: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
print(f"Parámetros congelados: {sum([tf.keras.backend.count_params(w) for w in base_model.weights]):,}")

# 5. Compilar el modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

# 6. Configurar callbacks para un entrenamiento óptimo
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_resnet50_cifar10.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 7. Entrenar el modelo (Transfer Learning)
print("\n" + "="*50)
print("FASE 1: TRANSFER LEARNING (capas congeladas)")
print("="*50)

history_phase1 = model.fit(
    x_train, y_train,
    batch_size=gpu_config['batch_size'],
    epochs=15,  # Pocas epochs porque usamos pesos pre-entrenados
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 8. Fine-tuning: Descongelar algunas capas superiores para ajuste fino
print("\n" + "="*50)
print("FASE 2: FINE TUNING (descongelando capas superiores)")
print("="*50)

# Descongelar las últimas capas del modelo base
base_model.trainable = True

# Congelar todas las capas excepto las últimas 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompilar con learning rate más bajo para fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # LR más bajo
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

print(f"Parámetros entrenables después del fine-tuning: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

# Continuar entrenamiento con fine-tuning
history_phase2 = model.fit(
    x_train, y_train,
    batch_size=gpu_config['batch_size_fine_tune'],  # Batch size más pequeño para fine-tuning
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 9. Evaluación final
print("\n" + "="*50)
print("EVALUACIÓN FINAL")
print("="*50)

test_loss, test_accuracy, test_top3_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Pérdida en test: {test_loss:.4f}")
print(f"Precisión en test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Top-3 Accuracy en test: {test_top3_accuracy:.4f} ({test_top3_accuracy*100:.2f}%)")

# 10. Combinar historiales y visualizar
def plot_training_history(history1, history2):
    """Visualiza el historial de entrenamiento de ambas fases"""
    
    # Combinar historiales
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])
    
    plt.figure(figsize=(15, 5))
    
    # Gráfico de precisión
    plt.subplot(1, 3, 1)
    plt.plot(epochs, acc, 'bo-', label='Precisión entrenamiento')
    plt.plot(epochs, val_acc, 'ro-', label='Precisión validación')
    plt.axvline(x=phase1_end, color='green', linestyle='--', label='Inicio Fine-tuning')
    plt.title('Precisión del modelo')
    plt.xlabel('Epochs')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de pérdida
    plt.subplot(1, 3, 2)
    plt.plot(epochs, loss, 'bo-', label='Pérdida entrenamiento')
    plt.plot(epochs, val_loss, 'ro-', label='Pérdida validación')
    plt.axvline(x=phase1_end, color='green', linestyle='--', label='Inicio Fine-tuning')
    plt.title('Pérdida del modelo')
    plt.xlabel('Epochs')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de learning rate
    plt.subplot(1, 3, 3)
    lr_phase1 = [0.001] * len(history1.history['lr'])
    lr_phase2 = history2.history['lr']
    all_lr = lr_phase1 + lr_phase2
    plt.plot(epochs, all_lr, 'go-', label='Learning Rate')
    plt.axvline(x=phase1_end, color='green', linestyle='--', label='Inicio Fine-tuning')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history_phase1, history_phase2)

# 11. Realizar predicciones y mostrar ejemplos
print("\n" + "="*50)
print("PREDICCIONES DE MUESTRA")
print("="*50)

# Predecir en una muestra de test
sample_size = 16
sample_indices = np.random.choice(len(x_test), sample_size, replace=False)
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]

predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(sample_labels, axis=1)
confidence_scores = np.max(predictions, axis=1)

# Mostrar resultados
print("Predicciones (con confianza):")
for i in range(min(10, sample_size)):
    pred_class = class_names[predicted_classes[i]]
    true_class = class_names[true_classes[i]]
    confidence = confidence_scores[i]
    status = "✅" if predicted_classes[i] == true_classes[i] else "❌"
    print(f"{status} Predicción: {pred_class} ({confidence:.3f}) | Real: {true_class}")

# 12. Matriz de confusión
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predecir en todo el conjunto de test
all_predictions = model.predict(x_test)
all_predicted_classes = np.argmax(all_predictions, axis=1)
all_true_classes = np.argmax(y_test, axis=1)

# Matriz de confusión
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_true_classes, all_predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión - ResNet50 en CIFAR-10')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Reporte de clasificación
print("\nReporte de clasificación detallado:")
print(classification_report(all_true_classes, all_predicted_classes, 
                          target_names=class_names, digits=4))

print(f"\nEntrenamiento completado! Mejor precisión: {test_accuracy*100:.2f}%")
print("Modelo guardado como 'best_resnet50_cifar10.h5'")