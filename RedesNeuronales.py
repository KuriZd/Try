import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Inicializar MediaPipe Face Mesh (para la cara)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Inicializar OpenCV para capturar video
cap = cv2.VideoCapture(0)

# Cargar el modelo entrenado previamente (sustituir con tu propio modelo entrenado)
model = tf.keras.models.load_model('mi_modelo_entrenado.h5')  # Cargar el modelo previamente entrenado

while cap.isOpened():
    ret, frame = cap.read()
    
    # Convertir el frame de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar el frame con MediaPipe para obtener los landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            # Extraer coordenadas de los landmarks (usando la cara como ejemplo)
            landmark_coords = []
            for landmark in landmarks.landmark:
                landmark_coords.append([landmark.x, landmark.y, landmark.z])

            # Convertir a un array de numpy y aplanar para que sea un vector
            landmark_coords = np.array(landmark_coords).flatten()  # Aplanar a un vector

            # Normalizar las coordenadas (opcional, dependiendo de cómo entrenaste tu modelo)
            # landmark_coords = (landmark_coords - np.mean(landmark_coords)) / np.std(landmark_coords)

            # Hacer predicción
            prediction = model.predict(landmark_coords.reshape(1, -1))  # Redimensionar para que coincida con el formato de entrada
            emotion = np.argmax(prediction)  # Obtener la clase de emoción con mayor probabilidad
            
            # Mostrar la emoción en la pantalla (esto depende de cómo etiquetaste las emociones en tu entrenamiento)
            emociones = ["Feliz", "Triste", "Enojado", "Sorpresa", "Miedo", "Neutral"]
            cv2.putText(frame, f"Emocion: {emociones[emotion]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Mostrar el frame procesado
    cv2.imshow('Predicción de emociones en tiempo real', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
