import os
import re
import cv2
import numpy as np
import insightface
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# Strict regex for safe filenames: only alphanumeric and underscores, 1-50 chars
SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{1,50}$')


def sanitize_name(name):
    """Sanitize a person name into a safe filesystem-friendly string.
    Raises ValueError if the result is empty or contains unsafe characters."""
    clean = "".join(c if c.isalnum() else "_" for c in name.strip())
    clean = re.sub(r'_+', '_', clean).strip('_')  # collapse multiple underscores
    if not clean or not SAFE_NAME_PATTERN.match(clean):
        raise ValueError(f"Invalid name after sanitization: '{clean}'. Use alphanumeric characters and underscores only (1-50 chars).")
    return clean


class FaceEngine:
    def __init__(self, embeddings_dir="embeddings", samples_dir="registration_samples", match_threshold=0.55):
        self.embeddings_dir = os.path.abspath(embeddings_dir)
        self.samples_dir = os.path.abspath(samples_dir)
        self.match_threshold = match_threshold

        # Parameters for quality checks (preserved from original)
        self.min_face_size = 100
        self.quality_threshold = 50
        self.min_sample_interval = 0.2  # seconds between samples

        # Initialize directories
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Thread safety lock for model inference
        self._lock = threading.Lock()

        logger.info("Creating FaceAnalysis object")

        self.model = insightface.app.FaceAnalysis(
            name="buffalo_s",
            providers=providers
        )

        logger.info("FaceAnalysis object created")

        logger.info("Preparing InsightFace model...")

        self.model.prepare(
            ctx_id=ctx_id,
            det_size=(320, 320)
        )

        logger.info("InsightFace model initialized successfully.")

        # Load database
        self.database = {}
        self.load_database()

        # In-memory registration sessions
        self.sessions = {}

    # ------------------------------------------------------------------
    # Database operations
    # ------------------------------------------------------------------

    def load_database(self):
        """Loads all registered mean embeddings from the embeddings folder."""
        new_database = {}
        if not os.path.isdir(self.embeddings_dir):
            self.database = {}
            return

        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith("_mean.npy"):
                name = filename.replace("_mean.npy", "")
                filepath = os.path.join(self.embeddings_dir, filename)
                try:
                    emb = np.load(filepath)
                    norm = np.linalg.norm(emb)
                    if norm == 0:
                        logger.warning("Zero-norm embedding for '%s', skipping.", name)
                        continue
                    emb = emb / norm
                    new_database[name] = emb
                except Exception:
                    logger.exception("Failed to load embedding for '%s'.", name)

        self.database = new_database
        logger.info("Database loaded: %d registered users.", len(self.database))

    # ------------------------------------------------------------------
    # Quality checks (preserved from original register_angle_face.py)
    # ------------------------------------------------------------------

    def check_face_quality(self, face_img):
        """Checks the quality of the cropped face image based on sharpness, brightness, and size."""
        if face_img is None or face_img.size == 0:
            return False, 0.0, "No face pixels detected."

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        if sharpness < self.quality_threshold:
            return False, sharpness, "Face image is too blurry."

        if brightness < 30:
            return False, sharpness, "Lighting is too dark. Please improve lighting."

        if brightness > 220:
            return False, sharpness, "Lighting is too bright. Please reduce lighting."

        if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
            return False, sharpness, "Face is too far away. Please move closer to the camera."

        return True, sharpness, "Success"

    # ------------------------------------------------------------------
    # Distance computation (preserved from original face_recognition.py)
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_distance(a, b):
        """Calculates the cosine distance between two normalized vectors."""
        return float(1.0 - np.dot(a, b))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def start_registration(self, name):
        """Starts a registration session for a given person name."""
        clean_name = sanitize_name(name)
        self.sessions[clean_name] = {
            "samples": [],  # list of tuples: (embedding, sharpness, face_img)
            "last_capture_time": 0,
            "last_face_key": None
        }
        logger.info("Registration session started for '%s'.", clean_name)
        return clean_name

    def process_registration_frame(self, name, frame, current_time):
        """
        Processes a single frame for user registration.
        Returns: (success_bool, message, current_count)
        """
        if name not in self.sessions:
            return False, "Registration session not found or expired.", 0

        session = self.sessions[name]

        with self._lock:
            faces = self.model.get(frame)

        if not faces:
            return False, "No face detected in the frame.", len(session["samples"])

        if len(faces) > 1:
            return False, "Multiple faces detected. Make sure only one person is in front of the camera.", len(session["samples"])

        # Process the single face
        face = faces[0]
        bbox = face.bbox.astype(int)

        # Ensure bounding box is within frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(w, bbox[2])
        y2 = min(h, bbox[3])
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
        logger.debug("Registration frame captured for '%s': %d/5 (sharpness=%.1f).", name, count, sharpness)
        return True, f"Captured sample {count}/5", count

    def finish_registration(self, name):
        """
        Finishes the registration by selecting the best 5 samples,
        saving them and creating the mean embedding.
        """
        if name not in self.sessions:
            return False, "Registration session not found or expired."

        session = self.sessions.pop(name)
        samples = session["samples"]

        if len(samples) < 1:
            return False, "No valid samples were captured."

        # Sort samples by sharpness (highest score first)
        samples.sort(key=lambda x: x[1], reverse=True)
        best_samples = samples[:5]

        # Extract embeddings and cropped images
        embeddings = [item[0] for item in best_samples]
        face_imgs = [item[2] for item in best_samples]

        try:
            # Save mean embedding
            mean_emb = np.mean(embeddings, axis=0)
            np.save(os.path.join(self.embeddings_dir, f"{name}_mean.npy"), mean_emb)

            # Save individual embeddings
            for i, emb in enumerate(embeddings):
                np.save(os.path.join(self.embeddings_dir, f"{name}_{i}.npy"), emb)

            # Save registration images
            for i, face_img in enumerate(face_imgs):
                cv2.imwrite(os.path.join(self.samples_dir, f"{name}_{i}.jpg"), face_img)
        except Exception:
            logger.exception("Failed to save embeddings for '%s'.", name)
            return False, "Failed to save embeddings to disk."

        # Reload database
        self.load_database()

        logger.info("Registration complete for '%s': %d embeddings saved.", name, len(embeddings))
        return True, f"Registration complete for {name}!"

    def cancel_registration(self, name):
        """Cancels a registration session and cleans up."""
        removed = self.sessions.pop(name, None)
        if removed is not None:
            logger.info("Registration cancelled for '%s'.", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Recognition (preserved from original face_recognition.py)
    # ------------------------------------------------------------------

    def recognize_faces(self, frame):
        """
        Performs face recognition on the input frame.
        Draws bounding boxes and labels directly on the frame.
        Returns: (annotated_frame, list_of_detected_faces)
        """
        with self._lock:
            faces = self.model.get(frame)

        detected_faces = []
        annotated_frame = frame.copy()

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

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
                "distance": round(best_distance, 4),
                "confidence": round(max(0.0, 1.0 - best_distance), 4)
            })

            # Draw bounding box and text on the annotated frame
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)

            cv2.rectangle(
                annotated_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color, 2
            )

            label = f"{best_name} ({best_distance:.2f})"
            cv2.putText(
                annotated_frame, label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        return annotated_frame, detected_faces

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def get_registered_users(self):
        """Returns metadata for all registered users."""
        users = []
        if not os.path.isdir(self.embeddings_dir):
            return users

        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith("_mean.npy"):
                name = filename.replace("_mean.npy", "")
                mean_path = os.path.join(self.embeddings_dir, filename)

                # Count individual embeddings
                emb_count = 0
                for f in os.listdir(self.embeddings_dir):
                    if f.startswith(f"{name}_") and f != f"{name}_mean.npy" and f.endswith(".npy"):
                        parts = f.replace(f"{name}_", "").replace(".npy", "")
                        if parts.isdigit():
                            emb_count += 1

                # Get registration date from file modified time
                try:
                    mtime = os.path.getmtime(mean_path)
                    reg_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                except OSError:
                    reg_date = "Unknown"

                users.append({
                    "name": name,
                    "embeddings_count": emb_count,
                    "registration_date": reg_date
                })

        users.sort(key=lambda x: x["name"].lower())
        return users

    def delete_user(self, name):
        """Deletes all embeddings and registration sample images for the specified user."""
        clean_name = sanitize_name(name)
        deleted_files = 0

        # Delete embeddings (only files matching the exact pattern)
        if os.path.isdir(self.embeddings_dir):
            for filename in os.listdir(self.embeddings_dir):
                if filename.startswith(f"{clean_name}_") and filename.endswith(".npy"):
                    try:
                        os.remove(os.path.join(self.embeddings_dir, filename))
                        deleted_files += 1
                    except OSError:
                        logger.exception("Failed to delete embedding file: %s", filename)

        # Delete registration images
        if os.path.isdir(self.samples_dir):
            for filename in os.listdir(self.samples_dir):
                if filename.startswith(f"{clean_name}_") and filename.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(self.samples_dir, filename))
                        deleted_files += 1
                    except OSError:
                        logger.exception("Failed to delete sample image: %s", filename)

        # Reload database
        self.load_database()

        logger.info("Deleted user '%s': %d files removed.", clean_name, deleted_files)
        return deleted_files > 0
