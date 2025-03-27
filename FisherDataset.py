import cv2 as cv
import numpy as np
import os

# Ruta del dataset de imágenes
dataSet = 'Faces'
faces = os.listdir(dataSet)
print(faces)

# Inicialización de etiquetas y datos
labels = []
facesData = []
label = 0

# Recorriendo cada persona (carpeta) y sus imágenes
for face in faces:
    facePath = os.path.join(dataSet, face)
    
    # Verificar si la ruta es un directorio (carpeta)
    if os.path.isdir(facePath):
        for faceName in os.listdir(facePath):
            faceImagePath = os.path.join(facePath, faceName)
            
            # Verificar si la ruta es un archivo (imagen)
            if os.path.isfile(faceImagePath):
                labels.append(label)
                facesData.append(cv.imread(faceImagePath, 0))
        
        label = label + 1

# Creación y entrenamiento del reconocedor facial
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))

# Guardado del modelo entrenado
faceRecognizer.write('laloFisherFace.xml')
