import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque, Counter
import google.protobuf

# 🔧 Fix protobuf para MediaPipe
if not hasattr(google.protobuf.message_factory, "GetMessageClass"):
    google.protobuf.message_factory.GetMessageClass = lambda descriptor: descriptor._concrete_class

# ================= CONFIG =================
VIDEO_PATH = r"C:\Users\rodri\Desktop\ab.mp4"
MODEL_PATH = r"C:\Users\rodri\Desktop\best_model_resnet50.h5"
IMG_SIZE = 160
SMOOTHING_FRAMES = 10

# ================= INICIALIZACIÓN =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

print("🧠 Cargando modelo...")
model = load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")

num_classes = model.output_shape[-1]
CLASSES = [chr(i) for i in range(65, 65 + num_classes)]
print(f"Clases detectadas ({num_classes}):", CLASSES)

pred_queue = deque(maxlen=SMOOTHING_FRAMES)

# ================= FUNCIONES =================
def predict_letter(frame, bbox):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None, 0
    hand_img = frame[y:y+h, x:x+w]
    if hand_img.size == 0:
        return None, 0

    # Resize y preprocesamiento
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(hand_img, axis=0)
    img_array = preprocess_input(img_array)

    # Predicción
    preds = model.predict(img_array, verbose=0)
    letter = CLASSES[np.argmax(preds)]
    conf = float(np.max(preds))
    return letter, conf

# ================= PROCESAMIENTO VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error al abrir el video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"⚠️ Se leyó hasta el frame {frame_count}")
        break

    frame_count += 1
    try:
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_coords) * w) - 20
                x_max = int(max(x_coords) * w) + 20
                y_min = int(min(y_coords) * h) - 20
                y_max = int(max(y_coords) * h) + 20

                # Limites del frame
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # ...
            letter, conf = predict_letter(frame, (x_min, y_min, x_max - x_min, y_max - y_min))
            if letter and conf >= 0.10:  # <- solo mostrar si la confianza es >= 85%
                pred_queue.append(letter)
                smooth_letter = Counter(pred_queue).most_common(1)[0][0]

                # Dibujar resultados
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'{smooth_letter} ({conf:.2f})', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)


        # Mostrar frame
        cv2.imshow("Predicción ASL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹ Salida manual (q presionado)")
            break

    except Exception as e:
        print(f"❌ Error en frame {frame_count}: {e}")
        continue  # No cerrar todo el video, seguir con el siguiente frame

cap.release()
hands.close()
cv2.destroyAllWindows()
print(f"✅ Finalizado. Se procesaron {frame_count} frames.")
