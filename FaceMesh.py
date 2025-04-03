import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

# Lista de índices de landmarks específicos (ojos, boca y nariz)
selected_points = [33, 133, 362, 263, 61, 291, 1, 168, 6]  # Ojos, boca y nariz

# Función de distancia entre puntos
def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

puntos_oscar = {
    33: (150, 120),  # Ojo izquierdo
    133: (200, 120),  # Ojo derecho
    362: (100, 200),  # Boca izquierda
    263: (250, 200),  # Boca derecha
    61: (120, 160),  # Nariz
    291: (230, 160),  # Nariz
    1: (175, 80),  # Punta de la nariz
    168: (180, 140),  # Puente de la nariz
    6: (160, 140),  # Costado de la nariz
}

def es_oscar(puntos_detectados):
    umbral = 50  # Umbral de distancia que define si las características coinciden
    distancias = []
    for idx in selected_points:
        if idx in puntos_detectados and idx in puntos_oscar:
            distancias.append(distancia(puntos_oscar[idx], puntos_detectados[idx]))
    
    # Compara las distancias promedio entre puntos de Oscar y los puntos detectados
    distancia_promedio = np.mean(distancias) if distancias else float('inf')
    return distancia_promedio < umbral  # Si la distancia promedio es menor que el umbral, se reconoce como Oscar

def dibujar_linea_referencia(frame):
    altura, anchura, _ = frame.shape
    y_linea = int(altura / 2) 
    cv2.line(frame, (0, y_linea), (anchura, y_linea), (255, 0, 0), 2)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Dibujar la línea de referencia
    dibujar_linea_referencia(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            puntos_detectados = {}

            # Detectar puntos faciales
            for idx in selected_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos_detectados[idx] = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Dibuja el punto en verde

            
            if 1 in puntos_detectados:
                x_nariz, y_nariz = puntos_detectados[1]
                if abs(y_nariz - (frame.shape[0] // 2)) < 50:  # Umbral de alineación
                    # Si la nariz está cerca de la línea de referencia, mostramos el nombre
                    if es_oscar(puntos_detectados):
                        cv2.putText(frame, "Oscar", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Desconocido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar el frame con los puntos, la línea de referencia y el nombre
    cv2.imshow('PuntosFacialesMediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
