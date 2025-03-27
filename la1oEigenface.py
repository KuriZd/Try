import cv2
import os

# Inicializar el reconocedor EigenFace
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('la1oEigenface.xml')  # Cargar modelo entrenado

# Inicializar el clasificador de rostros (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Diccionario de nombres (ejemplo: {0: "Juan", 1: "Maria"})
faces = {
    0: "Juan",
    1: "Maria"
}

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_copy = gray.copy()

    # Detectar rostros
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        face_roi = gray_copy[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100), interpolation=cv2.INTER_CUBIC)

        # Predecir la identidad del rostro
        label, confidence = face_recognizer.predict(face_roi)

        # Umbral de confianza (ajustable)
        if confidence < 2800:  # Si la confianza es alta (valor bajo)
            name = faces.get(label, "Desconocido")
            color = (0, 255, 0)  # Verde: reconocido
        else:
            name = "Desconocido"
            color = (0, 0, 255)  # Rojo: no reconocido

        # Dibujar rectángulo y texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Reconocimiento Eigenfaces', frame)

    # Salir con 'ESC'
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()