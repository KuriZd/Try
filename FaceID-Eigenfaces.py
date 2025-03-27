import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Eigenfaces:
    def __init__(self, data_path, test_size=0.2, n_components=25):
        self.data_path = data_path
        self.test_size = test_size
        self.n_components = n_components
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, whiten=True)
        
    def load_dataset(self):
        faces = []
        labels = []
        people = [person for person in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, person))]
        
        print(f"Personas encontradas en el conjunto de datos: {people}")
        
        for i, person in enumerate(people):
            person_path = os.path.join(self.data_path, person)
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))  # Normalizar tamaño
                    faces.append(img.flatten())
                    labels.append(i)
        
        return np.array(faces), np.array(labels), people
    
    def train(self):
        X, y, people = self.load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        
        # Escalar y aplicar PCA
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.pca.fit(X_train_scaled)
        
        # Transformar datos
        X_train_pca = self.pca.transform(X_train_scaled)
        self.model.fit(X_train_pca, y_train)
        
        # Evaluación (opcional)
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        accuracy = self.model.score(X_test_pca, y_test)
        print(f"Precisión del modelo: {accuracy:.2f}")
        
        return people
    
    def recognize_face(self, frame, people):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100)).flatten()
            face_scaled = self.scaler.transform([face_roi])
            face_pca = self.pca.transform(face_scaled)
            pred = self.model.predict(face_pca)[0]
            confidence = np.max(self.model.predict_proba(face_pca))
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{people[pred]} ({confidence:.2f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

# Uso del sistema
if __name__ == "__main__":
    DATA_PATH = "Faces"  # Actualiza esta ruta a donde tienes tus imágenes
    eigenfaces = Eigenfaces(DATA_PATH)
    people = eigenfaces.train()
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Reconocer la cara y mostrar el resultado en el frame
        frame = eigenfaces.recognize_face(frame, people)
        
        # Mostrar el mensaje de salida
        cv2.putText(frame, "Presiona 'q' para salir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Mostrar la ventana con el reconocimiento
        cv2.imshow('Eigenface Recognition', frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
