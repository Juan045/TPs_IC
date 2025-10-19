from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.image import resize_with_pad
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.data import AUTOTUNE

# (ajustá según tu estructura)
#dataset_dir = "../ASL_Alphabet_Dataset/asl_alphabet_train"
import os
dataset_dir = os.path.normpath("../ASL_Alphabet_Dataset/asl_alphabet_train")

# Parametros
img_height = 224
img_width = 224
batch_size = 32
val_split = 0.2   # 20% para validación
seed = 123

# Cargar datos de entrenamiento
train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=val_split,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),   
    batch_size=batch_size,
    pad_to_aspect_ratio=True
)

# Cargar datos de validación
val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=val_split,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width), 
    batch_size=batch_size,
    pad_to_aspect_ratio=True
)

# Aplicar resize con padding y normalización
def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

# Optimización con cache y prefetch
train_ds = train_ds.shuffle(500).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Verificar una muestra
for images, labels in train_ds.take(1):
    print(f"Batch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")