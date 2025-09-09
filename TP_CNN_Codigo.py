import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt


# 1. Cargar y normalizar los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizamos los valores de los píxeles en el rango [0,1].
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convertimos las etiquetas en one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)

import Arquitecturas as arch

# Creamos el modelo
model = arch.arquitectura()

# Compilamos con Adam .
# categorical_crossentropy se usa porque las clases son excluyentes (multiclase).
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary()) 

# Entrenamos con Adam
history_adam = model.fit(x_train, y_train,
                         validation_data=(x_test, y_test),
                         epochs=20, batch_size=64, verbose=1)


# Evaluamos el modelo en el conjunto de test.
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy con Adam: {acc:.4f}")

# -------------------------
# Creamos otro modelo idéntico para comparar con otro optimizador.
model_sgd = arch.arquitectura()

# Compilamos con SGD (Stochastic Gradient Descent).
model_sgd.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

# Entrenamos con SGD
history_sgd = model_sgd.fit(x_train, y_train,
                            validation_data=(x_test, y_test),
                            epochs=20, batch_size=64, verbose=1)

# Evaluamos el modelo entrenado con SGD
loss_sgd, acc_sgd = model_sgd.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy con SGD: {acc_sgd:.4f}")


plt.figure(figsize=(12,5))

# Gráfico de precisión en validación para Adam vs SGD
plt.subplot(1,2,1)
plt.plot(history_adam.history['val_accuracy'], label='Adam - Val Acc')
plt.plot(history_sgd.history['val_accuracy'], label='SGD - Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Precisión en Validación")
plt.legend()

# Gráfico de pérdida en validación para Adam vs SGD
plt.subplot(1,2,2)
plt.plot(history_adam.history['val_loss'], label='Adam - Val Loss')
plt.plot(history_sgd.history['val_loss'], label='SGD - Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Pérdida en Validación")
plt.legend()

plt.show()