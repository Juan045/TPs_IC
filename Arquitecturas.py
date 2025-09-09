from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


"""
# Descripción:
# arquitectura_1() implementa una red neuronal convolucional con tres bloques de convolución y pooling,
# seguida de capas densas y dropout, diseñada para extraer y combinar características de imágenes
# de tamaño 32x32x3, optimizada para tareas de clasificación como CIFAR-10.
"""
def arquitectura_1():
    model = Sequential()

    # Bloque 1: detección de características simples (bordes, líneas)
    # Conv2D aplica filtros convolucionales (kernels) que extraen patrones de la imagen.
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
    # BatchNormalization estabiliza la activación y acelera el entrenamiento.
    model.add(BatchNormalization())
    # Segunda convolución para refinar características en este nivel.
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    # MaxPooling reduce la dimensión espacial (submuestreo), manteniendo lo más relevante.
    model.add(MaxPooling2D((2,2)))
    # Dropout apaga aleatoriamente neuronas para evitar overfitting.
    model.add(Dropout(0.25))

    # Bloque 2: características intermedias (texturas, formas simples)
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    # Bloque 3: características más complejas (partes de objetos)
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    # Clasificación final
    # Flatten transforma los mapas de características en un vector 1D.
    model.add(Flatten())
    # Dense con ReLU combina las características extraídas para la clasificación.
    model.add(Dense(256, activation='relu'))
    # Dropout adicional para evitar sobreajuste en la capa densa.
    model.add(Dropout(0.5))
    # Capa de salida: 10 neuronas (una por clase de CIFAR-10), activación softmax para obtener probabilidades.
    model.add(Dense(10, activation='softmax'))

    return model