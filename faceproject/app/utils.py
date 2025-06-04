import cv2
import numpy as np
import joblib
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from django.conf import settings
import os
from sklearn.neighbors import KNeighborsClassifier
from .models import FaceImage, Person

class FaceRecognizer:
    _instances = {}
    
    def __init__(self, session_key):
        self.session_key = session_key
        self.running = False
        self.cap = None
        self.expected_embedding_size = 512
        self.confidence_threshold = 0.6
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._initialize_components()
        FaceRecognizer._instances[session_key] = self

    def _initialize_components(self):
        # Initialize face detection and recognition models
        self.mtcnn = MTCNN(
            keep_all=True,
            thresholds=[0.7, 0.7, 0.8],
            device=self.device
        )
        self.resnet = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=self.device
        ).eval()
        
        # Load KNN classifier
        self.model_path = os.path.join(settings.BASE_DIR, 'app', 'face_classifier.pkl')
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = KNeighborsClassifier(n_neighbors=3, metric='cosine')
            self.train_model()

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb_img)
        return boxes.astype(int) if boxes is not None else []

    def process_face(self, face_img):
        try:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (160, 160))
            face_tensor = torch.from_numpy(face_img).permute(2,0,1).float().to(self.device)
            face_tensor = (face_tensor - 127.5) / 128.0
            embedding = self.resnet(face_tensor.unsqueeze(0))
            return embedding.detach().cpu().numpy().flatten()
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            return None

    def train_model(self):
        try:
            # Get all face images with their embeddings
            face_images = FaceImage.objects.all().select_related('person')
            
            X, y = [], []
            for img in face_images:
                try:
                    embedding = np.frombuffer(img.embedding, dtype=np.float32)
                    if len(embedding) == self.expected_embedding_size:
                        X.append(embedding)
                        y.append(img.person.id)
                except Exception as e:
                    print(f"Error processing image {img.id}: {str(e)}")
                    continue

            if not X:
                print("No valid embeddings found for training")
                return

            # Dynamically set n_neighbors
            n_neighbors = min(3, len(X))
            if len(X) < 3:
                print(f"Warning: Using n_neighbors={n_neighbors} (only {len(X)} samples available)")

            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric='cosine',
                weights='distance'
            )
            self.model.fit(X, y)
            joblib.dump(self.model, self.model_path)
            print(f"Model trained with {len(X)} samples (n_neighbors={n_neighbors})")

        except Exception as e:
            print(f"Training failed: {str(e)}")

            
    @classmethod
    def retrain_global_model(cls):
        for instance in cls._instances.values():
            instance.train_model()

    def start_camera(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open video device")

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            del FaceRecognizer._instances[self.session_key]

    @classmethod
    def stop_all(cls):
        for instance in list(cls._instances.values()):
            instance.stop_camera()

    def get_frame(self):
        if not self.running or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        # Detect faces
        faces = self.detect_faces(frame)
        id_to_name = {p.id: p.name for p in Person.objects.all()}

        for box in faces:
            try:
                x1, y1, x2, y2 = box
                
                x1 =max(x1,0)
                y1 = max(y1,0)
                x2 = min(x2, frame.shape[1])
                y2 = min(y2, frame.shape[1])


                #  the box has a positive area
                if x2 <= x1 or y2<=y1:
                    continue

                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue
                # Process face
                embedding = self.process_face(face_img)
                
                if embedding is not None and hasattr(self.model, "predict"):
                    person_id = self.model.predict([embedding])[0]
                    confidence = self.model.predict_proba([embedding])[0].max()
                    
                    if confidence > self.confidence_threshold:
                        person_name = id_to_name.get(person_id, "Unknown")
                    else:
                        person_name = "Unknown"
                    
                    # Draw annotations
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, person_name, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"Face processing error: {str(e)}")

        return frame

    def generate_frames(self):
        self.start_camera()
        try:
            while self.running:
                frame = self.get_frame()
                if frame is None:
                    break
                
                # Convert to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            self.stop_camera()