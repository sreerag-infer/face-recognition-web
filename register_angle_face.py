import cv2
import numpy as np
import insightface
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import time


class FaceRegister:
    def __init__(self):
        # Initialize InsightFace model
        self.model = insightface.app.FaceAnalysis(
            name='buffalo_s',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.model.prepare(ctx_id=0, det_size=(320, 320))

        # Parameters
        self.min_face_size = 100
        self.quality_threshold = 50
        self.min_sample_interval = 0.2

        # Create directories
        os.makedirs("embeddings", exist_ok=True)
        os.makedirs("registration_samples", exist_ok=True)
        
    def check_face_quality(self, face_img):
        """Quality score based on sharpness."""
        if face_img.size == 0:
            return False, 0

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        if sharpness < self.quality_threshold:
            return False, sharpness
        if brightness < 30 or brightness > 220:
            return False, sharpness
        if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
            return False, sharpness

        return True, sharpness

    def get_person_name(self):
        """Ask user for a name."""
        root = tk.Tk()
        root.withdraw()
        name = simpledialog.askstring("Name Input", "Enter person's name:")
        if not name:
            return None
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        return clean_name

    def select_video_file(self):
        """Open file dialog (supports .webm)."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        try:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")
                ]
            )
            return file_path
        except Exception as e:
            print(f"File selection error: {str(e)}")
            return None
        finally:
            try:
                root.destroy()
            except:
                pass

    def collect_best_embeddings(self, source):
        """
        Collect ALL embeddings, track quality, return best 5.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video source: {source}")
            return []

        # Store (embedding, sharpness_score, timestamp, face_img_path)
        all_samples = []
        last_face_key = None
        last_capture_time = 0

        feedback_window = "Auto Registration - Press ESC to stop"
        frame_skip = 3
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if source != 0 and frame_count % frame_skip != 0:
                    continue

                faces = self.model.get(frame)
                current_time = time.time()

                if not faces:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    largest = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    )
                    bbox = largest.bbox.astype(int)
                    face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    is_ok, sharpness = self.check_face_quality(face_img)
                    current_key = tuple(bbox // 10)
                    time_ok = current_time - last_capture_time > self.min_sample_interval

                    if is_ok and time_ok and current_key != last_face_key:
                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        sample_path = os.path.join("registration_samples", f"auto_{timestamp}.jpg")
                        cv2.imwrite(sample_path, face_img)

                        # Store with quality score
                        all_samples.append((largest.embedding, sharpness, timestamp, sample_path))

                        last_face_key = current_key
                        last_capture_time = current_time

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, "CAPTURED!", (bbox[0], bbox[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        color = (0, 255, 0) if is_ok else (0, 0, 255)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                cv2.putText(frame, f"Samples: {len(all_samples)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(feedback_window, frame)
                if cv2.waitKey(10) == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        # --- Select BEST 5 embeddings ---
        if len(all_samples) == 0:
            return []

        # Sort by sharpness (quality), take top 5
        all_samples.sort(key=lambda x: x[1], reverse=True)  # highest sharpness first
        best_5 = all_samples[:5]

        print(f"\nTotal collected: {len(all_samples)}, Best 5 selected.")
        for i, (emb, score, ts, path) in enumerate(best_5):
            print(f"  Best #{i+1}: sharpness={score:.1f}")

        return [emb for emb, _, _, _ in best_5]  # return only embeddings

    def save_person_embeddings(self, name, embeddings):
        """Save exactly 5 embeddings + mean."""
        if len(embeddings) == 0:
            return False

        # Mean embedding
        mean_emb = np.mean(embeddings, axis=0)
        np.save(f"embeddings/{name}_mean.npy", mean_emb)

        # Individual embeddings
        for i, emb in enumerate(embeddings):
            np.save(f"embeddings/{name}_{i}.npy", emb)

        print(f"Saved {name}: mean.npy + {len(embeddings)} individual embeddings")
        return True

    def register_from_source(self, source_type):
        # Get name first
        name = self.get_person_name()
        if not name:
            print("No name provided, skipping.")
            return

        if source_type == "camera":
            source = 0
            msg = "Live Camera"
        elif source_type == "video":
            source = self.select_video_file()
            if not source:
                return
            msg = f"Video: {source}"
        else:
            return

        print(f"\n=== Register {name} from {msg} ===")
        print("Press ESC to stop collecting.")

        # Collect best 5 embeddings
        best_embeddings = self.collect_best_embeddings(source)
        if len(best_embeddings) == 0:
            print("No good embeddings collected.")
            return

        # Save
        self.save_person_embeddings(name, best_embeddings)

    def run_registration(self):
        print("=== Best 5 Embeddings Registration ===")
        print("1. Register from live camera")
        print("2. Register from video file (.mp4, .avi, .webm, etc.)")
        print("3. Exit")

        while True:
            choice = input("\nSelect option (1/2/3): ").strip()

            if choice == '1':
                self.register_from_source("camera")
            elif choice == '2':
                self.register_from_source("video")
            elif choice == '3':
                break
            else:
                print("Invalid choice, please enter 1, 2, or 3")


if __name__ == "__main__":
    register = FaceRegister()
    register.run_registration()