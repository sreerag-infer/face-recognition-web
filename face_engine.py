import os
import cv2
import numpy as np
import insightface
import threading
from datetime import datetime

class FaceEngine:
    def __init__(self, embeddings_dir="embeddings", samples_dir="registration_samples", match_threshold=0.55):
        self.embeddings_dir = embeddings_dir
        self.samples_dir = samples_dir
        self.match_threshold = match_threshold
        
        # Parameters for quality checks
        self.min_face_size = 100
        self.quality_threshold = 50
        self.min_sample_interval = 0.2  # minimum interval in seconds between samples

        # Initialize directories
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Thread safety lock for model inference
        self.lock = threading.Lock()

        # Load InsightFace Model
        print("Initializing InsightFace model...")
        self.model = insightface.app.FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # 320x320 is standard and fast for web capture
        self.model.prepare(ctx_id=0, det_size=(320, 320))
        print("Model initialized successfully.")

        # Load database
        self.database = {}
        self.load_database()

        # In-memory registration sessions
        # Format: { username: { "samples": [...], "last_capture_time": float, "last_face_key": tuple } }
        self.sessions = {}

    def load_database(self):
        """Loads all registered mean embeddings from the embeddings folder."""
        new_database = {}
        if not os.path.exists(self.embeddings_dir):
            self.database = {}
            return

        for file in os.listdir(self.embeddings_dir):
            if file.endswith("_mean.npy"):
                name = file.replace("_mean.npy", "")
                try:
                    emb = np.load(os.path.join(self.embeddings_dir, file))
                    # Normalize
                    emb = emb / np.linalg.norm(emb)
                    new_database[name] = emb
                except Exception as e:
                    print(f"Error loading embedding for {name}: {str(e)}")
        
        self.database = new_database
        print(f"Database loaded. {len(self.database)} registered users.")

    def check_face_quality(self, face_img):
        """Checks the quality of the cropped face image based on sharpness, brightness, and size."""
        if face_img.size == 0:
            return False, 0, "No face pixels detected."

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        # Perform individual quality checks
        if sharpness < self.quality_threshold:
            return False, sharpness, f"Face image is too blurry."
        
        if brightness < 30:
            return False, sharpness, "Lighting is too dark. Please improve lighting."
        
        if brightness > 220:
            return False, sharpness, "Lighting is too bright. Please reduce lighting."
        
        if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
            return False, sharpness, f"Face is too far away. Please move closer to the camera."

        return True, sharpness, "Success"

    def cosine_distance(self, a, b):
        """Calculates the cosine distance between two vectors."""
        return 1 - np.dot(a, b)

    def start_registration(self, name):
        """Starts a registration session for a given person name."""
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        self.sessions[clean_name] = {
            "samples": [],  # list of tuples: (embedding, sharpness, face_img)
            "last_capture_time": 0,
            "last_face_key": None
        }
        return clean_name

    def process_registration_frame(self, name, frame, current_time):
        """
        Processes a single frame for user registration.
        Returns: (success_bool, message, current_count)
        """
        if name not in self.sessions:
            return False, "Registration session not found or expired.", 0

        session = self.sessions[name]
        
        with self.lock:
            faces = self.model.get(frame)

        if not faces:
            return False, "No face detected in the frame.", len(session["samples"])

        if len(faces) > 1:
            return False, "Multiple faces detected. Make sure only one person is in front of the camera.", len(session["samples"])

        # Process the single face
        face = faces[0]
        bbox = face.bbox.astype(int)
        
        # Ensure bounding box is within frame boundaries
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
        face_img = frame[y1:y2, x1:x2]

        is_ok, sharpness, reason = self.check_face_quality(face_img)
        if not is_ok:
            return False, reason, len(session["samples"])

        # Spatial key check (ensure face has moved slightly to collect varied angles)
        current_key = tuple(bbox // 10)
        time_ok = current_time - session["last_capture_time"] > self.min_sample_interval

        if not time_ok:
            return False, "Capturing frames too quickly. Please hold still or move slightly.", len(session["samples"])

        if current_key == session["last_face_key"]:
            return False, "Please move your head slightly to capture different angles.", len(session["samples"])

        # Capture valid sample
        session["samples"].append((face.embedding, sharpness, face_img.copy()))
        session["last_face_key"] = current_key
        session["last_capture_time"] = current_time

        count = len(session["samples"])
        return True, f"Captured sample {count}/5", count

    def finish_registration(self, name):
        """
        Finishes the registration by selecting the best 5 samples,
        saving them and creating the mean embedding.
        """
        if name not in self.sessions:
            return False, "Registration session not found or expired."

        session = self.sessions[name]
        samples = session["samples"]

        if len(samples) < 1:
            # Clean session
            self.sessions.pop(name, None)
            return False, "No valid samples were captured."

        # Sort samples by sharpness (highest score first)
        samples.sort(key=lambda x: x[1], reverse=True)
        best_samples = samples[:5]

        # Extract embeddings and cropped images
        embeddings = [item[0] for item in best_samples]
        face_imgs = [item[2] for item in best_samples]

        # Save embeddings
        # 1. Mean embedding
        mean_emb = np.mean(embeddings, axis=0)
        np.save(os.path.join(self.embeddings_dir, f"{name}_mean.npy"), mean_emb)

        # 2. Individual embeddings
        for i, emb in enumerate(embeddings):
            np.save(os.path.join(self.embeddings_dir, f"{name}_{i}.npy"), emb)

        # 3. Registration images
        for i, face_img in enumerate(face_imgs):
            cv2.imwrite(os.path.join(self.samples_dir, f"{name}_{i}.jpg"), face_img)

        # Clean session
        self.sessions.pop(name, None)

        # Reload database
        self.load_database()

        return True, f"Registration complete for {name}!"

    def cancel_registration(self, name):
        """Cancels a registration session and cleans up."""
        if name in self.sessions:
            self.sessions.pop(name, None)
            return True
        return False

    def recognize_faces(self, frame):
        """
        Performs face recognition on the input frame.
        Draws bounding boxes and labels directly on the frame.
        Returns: (annotated_frame, list_of_detected_faces)
        """
        with self.lock:
            faces = self.model.get(frame)

        detected_faces = []
        annotated_frame = frame.copy()

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            best_name = "Unknown"
            best_distance = 999.0

            # Compare against database
            for person_name, ref_embedding in self.database.items():
                distance = self.cosine_distance(embedding, ref_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_name = person_name

            # Apply match threshold
            if best_distance > self.match_threshold:
                best_name = "Unknown"

            detected_faces.append({
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "name": best_name,
                "distance": float(best_distance),
                "confidence": float(1.0 - best_distance) # confidence score helper
            })

            # Draw bounding box and text on the annotated frame
            # Match coloring: Green if recognized, Red if Unknown
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            
            # Draw bbox
            cv2.rectangle(
                annotated_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2
            )

            # Draw label text
            label = f"{best_name} (dist: {best_distance:.2f})"
            cv2.putText(
                annotated_frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return annotated_frame, detected_faces

    def get_registered_users(self):
        """Returns metadata for all registered users."""
        users = []
        if not os.path.exists(self.embeddings_dir):
            return users

        # Scan for mean files
        for file in os.listdir(self.embeddings_dir):
            if file.endswith("_mean.npy"):
                name = file.replace("_mean.npy", "")
                mean_path = os.path.join(self.embeddings_dir, file)
                
                # Count individual embeddings
                emb_count = 0
                for f in os.listdir(self.embeddings_dir):
                    if f.startswith(f"{name}_") and f != f"{name}_mean.npy" and f.endswith(".npy"):
                        # Ensure it matches name_{index}.npy
                        parts = f.replace(f"{name}_", "").replace(".npy", "")
                        if parts.isdigit():
                            emb_count += 1

                # Get registration date from file modified time
                mtime = os.path.getmtime(mean_path)
                reg_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

                users.append({
                    "name": name,
                    "embeddings_count": emb_count,
                    "registration_date": reg_date
                })

        # Sort by name
        users.sort(key=lambda x: x["name"])
        return users

    def delete_user(self, name):
        """Deletes all embeddings and registration sample images for the specified user."""
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        deleted_files = 0

        # Delete embeddings
        if os.path.exists(self.embeddings_dir):
            for file in os.listdir(self.embeddings_dir):
                if file.startswith(f"{clean_name}_") and file.endswith(".npy"):
                    try:
                        os.remove(os.path.join(self.embeddings_dir, file))
                        deleted_files += 1
                    except Exception as e:
                        print(f"Error deleting file {file}: {str(e)}")

        # Delete registration images
        if os.path.exists(self.samples_dir):
            for file in os.listdir(self.samples_dir):
                if file.startswith(f"{clean_name}_") and file.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(self.samples_dir, file))
                        deleted_files += 1
                    except Exception as e:
                        print(f"Error deleting file {file}: {str(e)}")

        # Reload database
        self.load_database()

        return deleted_files > 0
