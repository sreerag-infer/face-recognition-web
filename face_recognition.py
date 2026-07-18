import os
import cv2
import numpy as np
import insightface

# ==========================
# Configuration
# ==========================

EMBEDDING_FOLDER = "embeddings"
MATCH_THRESHOLD = 0.55
FRAME_SKIP = 3  # Process every 3rd frame

# ==========================
# Load InsightFace Model
# ==========================

app = insightface.app.FaceAnalysis(
    name="buffalo_s",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Reduce detection size for better speed
app.prepare(ctx_id=0, det_size=(224, 224))

# ==========================
# Load Registered Embeddings
# ==========================

database = {}

for file in os.listdir(EMBEDDING_FOLDER):
    if file.endswith("_mean.npy"):
        name = file.replace("_mean.npy", "")
        emb = np.load(os.path.join(EMBEDDING_FOLDER, file))
        emb = emb / np.linalg.norm(emb)
        database[name] = emb

print(f"Loaded {len(database)} registered people.")

# ==========================
# Cosine Distance
# ==========================

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

# ==========================
# Open Webcam
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

print("Press ESC to exit.")

frame_count = 0
last_results = []

# ==========================
# Recognition Loop
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Only run detection every FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:

        faces = app.get(frame)
        last_results = []

        for face in faces:

            bbox = face.bbox.astype(int)

            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            best_name = "Unknown"
            best_distance = 999

            for person_name, ref_embedding in database.items():

                distance = cosine_distance(embedding, ref_embedding)

                if distance < best_distance:
                    best_distance = distance
                    best_name = person_name

            if best_distance > MATCH_THRESHOLD:
                best_name = "Unknown"

            last_results.append(
                (bbox, best_name, best_distance)
            )

    # Draw previous detections (keeps video smooth)
    for bbox, name, distance in last_results:

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{name} ({distance:.2f})",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()