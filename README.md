# 🟥 Repositorio de los TPs de IC

Para ir hacia un trabajo practico cambiar de rama

## 🟧 Hacer una rama por cada TP

* El TP de CNN tiene su propia rama (TP_CNN)
* El TP de Algoritmos Geneticos tiene su propia rama (TP_AG)
* El TP del caso de estudio tiene su propia rama (TP_Caso)

>Cualquier cambio que no se desee poner en la rama del tp hasta que se este seguro de implementar el cambio, crearla con el nombre _branch/[Nombre del TP]_

## 🟩 Que se resuelve en la rama (TP_Caso)

Clasificacion del alfabeto dactilologico ASL (29 clases: A-Z mas `del`, `nothing` y `space`) mediante **transfer learning**, y su puesta a prueba sobre **video real**, fuera de las imagenes con las que se entreno.

### Dataset

[ASL (American Sign Language) Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) — Kaggle. 29 clases con imagenes de 200x200.

La ruta local se configura en la variable `dataset_dir` de cada script de entrenamiento.

### Archivos

| Archivo | Rol |
|---|---|
| `analysis.py` | Analisis exploratorio del dataset: balance de clases, tamaños, formatos y canales |
| `main.py` | Entrenamiento v1: ResNet50 congelada + cabeza densa, 224x224 con padding, evaluacion sobre el set de test |
| `mainv2.py` | Entrenamiento v2: ResNet50 en dos fases (base congelada + fine-tuning de las ultimas 30 capas), matriz de confusion y F1 macro |
| `mobilenet.py` | Misma estructura que v2 pero con MobileNetV2, como alternativa mas liviana |
| `video.py` / `video copy.py` | Inferencia sobre video: MediaPipe Hands localiza la mano, se recorta y se predice la letra frame a frame |
| `test.py` | Auxiliar: imprime el `summary()` de ResNet50 |

### Flujo de trabajo

Los scripts de entrenamiento guardan el mejor modelo via `ModelCheckpoint` en un archivo `.h5`. Los scripts de video **no llaman a los de entrenamiento**: son independientes y solo consumen ese `.h5` a traves de `load_model()`, por lo que hay que apuntar `MODEL_PATH` al archivo generado.

### Resultados y limitaciones

📹 **Video utilizado en la prueba final:** [Ver en YouTube](https://www.youtube.com/watch?v=6_gXiBe9y9A)

El modelo **reconoce correctamente los simbolos de la mano**: sobre imagenes del mismo dominio que el entrenamiento la clasificacion es acertada, y en video la deteccion y el recorte de la mano con MediaPipe funcionan bien.

La limitacion aparece al llevarlo a video: **el dataset no tiene suficiente diversidad de casos** (poca variedad de fondos, iluminacion, angulos, distancias y manos distintas), por lo que el modelo generaliza mal a condiciones que no vio durante el entrenamiento. Las predicciones erroneas en las pruebas de video reflejan esa falta de diversidad en los datos, no una falla de la arquitectura ni del pipeline de inferencia.
