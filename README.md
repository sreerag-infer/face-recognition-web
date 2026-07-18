# Face Registration & Recognition Web Application

A responsive web application that exposes a REST API and frontend interface for face enrollment and real-time face recognition. The system leverages your existing Python scripts, wrapping them inside a high-performance Flask application while preserving the core InsightFace matching logic, quality check parameters, and threshold constants.

## Features
- **Interactive Face Registration**: Enrolls users by streaming webcam frames. Automatically checks face image quality (sharpness, lighting, size) and selects the best 5 samples to generate a mean embedding profile.
- **Real-Time Recognition Feed**: Dynamically processes webcam frames, overlays face bounding boxes with match confidence labels, and details visibility events in a sidebar log.
- **Profile Management**: Displays active profiles, showing enrollment dates and embedding count metadata. Supports deleting profiles along with their mean embeddings, individual embeddings, and image samples.
- **Light/Dark Mode Dashboard**: Clean dashboard layouts with Bootstrap 5 templates, persisted preferences, and glassmorphism.
- **Production-Ready Deployment**: Includes configurations for Docker, Render blueprints, and Railway.

---

## Project Structure
```text
project/
│
├── app.py                       # Flask REST API endpoints and router
├── face_engine.py               # Encapsulates InsightFace, quality validation, matching, and database storage
├── register_angle_face.py       # (Original script preserved)
├── face_recognition.py          # (Original script preserved)
│
├── templates/                   # HTML5 Bootstrap 5 templates
│   ├── base.html                # Layout template with Navbar and Toast alerts
│   ├── index.html               # Home dashboard view
│   ├── register.html            # Registration stream and feedback UI
│   ├── recognize.html           # Live face scanner view
│   └── users.html               # Profiles management page
│
├── static/
│   └── css/
│       └── style.css            # Stylesheet (variables, animations, glass panels)
│
├── embeddings/                  # Saved .npy binary face embeddings (mean + individual)
├── registration_samples/        # Saved cropped face JPG samples of registered profiles
│
├── requirements.txt             # Python packages
├── Dockerfile                   # Docker container builder
├── Procfile                     # Web server command script
├── gunicorn.conf.py             # Gunicorn runtime worker limits
├── render.yaml                  # Render service blueprint
└── railway.json                 # Railway service configuration
```

---

## Local Setup

### 1. Prerequisite Dependencies
Make sure you have Python 3.10 (or 3.8+) installed. On Linux/Debian systems, you need CMake, compilers, and OpenCV graphical libraries:
```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake libgl1-mesa-glx libglib2.0-0
```

### 2. Installation
Clone or navigate to the directory:
```bash
# Create virtual environment
python -m venv reco
reco\Scripts\activate     # On Windows
source reco/bin/activate  # On macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 3. Launching
Run the Flask server locally:
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## Deployment

Deploying via Docker is highly recommended because of system compiling requirements for `insightface`.

### Render Deployment
1. Connect your repository to **Render**.
2. Click **New +** and select **Blueprint**.
3. Render will auto-discover the `render.yaml` configuration and deploy it as a Docker service with a persistent volume to store your enrolled embeddings.

### Railway Deployment
1. Create a new project on **Railway**.
2. Deploy from your GitHub repository.
3. Railway will read the `railway.json` file, auto-detect the `Dockerfile`, and deploy the web service.

---

## Backend REST APIs
- `GET /api/status`: Returns system statistics (user count, model runtime accelerator).
- `GET /api/users`: Returns JSON list of all enrolled profiles, embedding sizes, and registration dates.
- `POST /api/register/start`: Initializes a session for a user. Expects `{"name": "name"}`.
- `POST /api/register/frame`: Processes a base64 frame. Expects `{"name": "name", "image": "data:image/jpeg;base64,..."}`.
- `POST /api/register/finish`: Computes mean and saves embeddings. Expects `{"name": "name"}`.
- `POST /api/register/cancel`: Cancels session. Expects `{"name": "name"}`.
- `POST /api/recognize`: Scans frame, detects faces, checks matches, and returns an annotated image base64 stream. Expects `{"image": "data:image/jpeg;base64,..."}`.
- `DELETE /api/users/<name>`: Wipes the profile embeddings and photos from the filesystem.
