import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======== CONFIGURACIÓN ========
# Cambia esta ruta según donde hayas extraído el dataset
dataset_path = ""
# ================================


# --- Inicialización ---
class_names = []
image_counts = []
avg_widths = []
avg_heights = []
formats = []
channels = []


# --- 1. Conteo de imágenes por clase ---
for class_folder in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_folder)
    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    if len(images) == 0:
        continue

    widths, heights, img_formats, img_channels = [], [], [], []

    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                img_formats.append(img.format)
                img_channels.append(len(img.getbands()))
        except Exception as e:
            print(f"[AVISO] No se pudo procesar {img_path}: {e}")
            continue

    class_names.append(class_folder)
    image_counts.append(len(images))
    avg_widths.append(np.mean(widths))
    avg_heights.append(np.mean(heights))
    formats.extend(img_formats)
    channels.extend(img_channels)


# --- Crear DataFrame resumen ---
df_summary = pd.DataFrame({
    "Clase": class_names,
    "Cantidad de Imágenes": image_counts,
    "Ancho Promedio": avg_widths,
    "Alto Promedio": avg_heights
}).sort_values(by="Cantidad de Imágenes", ascending=False)

# Mostrar resumen en consola
print("\n=== Resumen por Clase ===")
print(df_summary.to_string(index=False))


# --- 3. Gráfico de balance de clases ---
plt.figure(figsize=(10, 6))
plt.bar(df_summary["Clase"], df_summary["Cantidad de Imágenes"])
plt.xticks(rotation=45, ha="right")
plt.title("Balance de Clases en el Dataset")
plt.xlabel("Clase")
plt.ylabel("Cantidad de Imágenes")
plt.tight_layout()
plt.show()


# --- 4. Análisis de formato y canales de color ---
format_counts = pd.Series(formats).value_counts()
channel_counts = pd.Series(channels).value_counts()

# Gráfico de formatos
plt.figure(figsize=(6, 4))
plt.bar(format_counts.index, format_counts.values)
plt.title("Distribución de Formatos de Imagen")
plt.xlabel("Formato")
plt.ylabel("Cantidad de Imágenes")
plt.tight_layout()
plt.show()

# Gráfico de canales de color
plt.figure(figsize=(6, 4))
plt.bar(channel_counts.index.astype(str), channel_counts.values)
plt.title("Distribución de Canales de Color")
plt.xlabel("Cantidad de Canales")
plt.ylabel("Número de Imágenes")
plt.tight_layout()
plt.show()
