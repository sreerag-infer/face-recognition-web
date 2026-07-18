import os
import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from face_engine import FaceEngine

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize Face Engine
# In production/deployment, we handle potential memory/CPU constraints gracefully
try:
    engine = FaceEngine()
except Exception as e:
    print(f"CRITICAL: Failed to initialize FaceEngine: {str(e)}")
    engine = None

def decode_base64_image(base64_str):
    """Decodes a base64 image string (data URI) into a CV2 BGR image."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        return None

def encode_image_to_base64(img):
    """Encodes a CV2 BGR image into a base64 JPEG data URI."""
    try:
        _, buffer = cv2.imencode('.jpg', img)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"Error encoding image to base64: {str(e)}")
        return ""

# ==========================
# Frontend Routes
# ==========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html')

@app.route('/users')
def users_page():
    return render_template('users.html')

# ==========================
# REST API Endpoints
# ==========================

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns the backend and face engine status."""
    if engine is None:
        return jsonify({
            "status": "error",
            "message": "FaceEngine is not initialized",
            "users_count": 0,
            "providers": []
        }), 500
    
    return jsonify({
        "status": "ok",
        "message": "System is running",
        "users_count": len(engine.database),
        "providers": engine.model.providers if hasattr(engine.model, 'providers') else ["CPUExecutionProvider"]
    })

# Add /api/people as an alias for compatibility
@app.route('/api/people', methods=['GET'])
@app.route('/api/users', methods=['GET'])
def get_users():
    """Returns list of registered users."""
    if engine is None:
        return jsonify({"users": [], "error": "FaceEngine not initialized"}), 500
    
    try:
        users = engine.get_registered_users()
        return jsonify({"users": users})
    except Exception as e:
        return jsonify({"users": [], "error": str(e)}), 500

@app.route('/api/register/start', methods=['POST'])
def register_start():
    """Starts a registration session."""
    if engine is None:
        return jsonify({"status": "error", "message": "FaceEngine not initialized"}), 500
        
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({"status": "error", "message": "Person name is required."}), 400
        
    # Check if user already exists
    clean_name = "".join(c if c.isalnum() else "_" for c in name)
    already_exists = clean_name in engine.database
    
    try:
        session_name = engine.start_registration(name)
        return jsonify({
            "status": "success",
            "name": session_name,
            "already_exists": already_exists,
            "message": f"Registration session started for {session_name}."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/register/frame', methods=['POST'])
def register_frame():
    """Processes a single frame for registration."""
    if engine is None:
        return jsonify({"success": False, "message": "FaceEngine not initialized"}), 500
        
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    image_base64 = data.get('image', '')
    
    if not name or not image_base64:
        return jsonify({"success": False, "message": "Name and image frame are required."}), 400
        
    img = decode_base64_image(image_base64)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image data."}), 400
        
    try:
        success, message, count = engine.process_registration_frame(name, img, time.time())
        return jsonify({
            "success": success,
            "message": message,
            "count": count
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/register/finish', methods=['POST'])
def register_finish():
    """Completes the registration process, saving npy files."""
    if engine is None:
        return jsonify({"status": "error", "message": "FaceEngine not initialized"}), 500
        
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({"status": "error", "message": "Person name is required."}), 400
        
    try:
        success, message = engine.finish_registration(name)
        if success:
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "message": message}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/register/cancel', methods=['POST'])
def register_cancel():
    """Cancels registration and cleans up."""
    if engine is None:
        return jsonify({"status": "error", "message": "FaceEngine not initialized"}), 500
        
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({"status": "error", "message": "Person name is required."}), 400
        
    try:
        engine.cancel_registration(name)
        return jsonify({"status": "success", "message": "Registration cancelled."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Runs face recognition on the uploaded frame."""
    if engine is None:
        return jsonify({"error": "FaceEngine not initialized"}), 500
        
    data = request.get_json() or {}
    image_base64 = data.get('image', '')
    
    if not image_base64:
        return jsonify({"error": "No image frame provided."}), 400
        
    img = decode_base64_image(image_base64)
    if img is None:
        return jsonify({"error": "Invalid image data."}), 400
        
    try:
        annotated_img, faces = engine.recognize_faces(img)
        annotated_base64 = encode_image_to_base64(annotated_img)
        return jsonify({
            "image": annotated_base64,
            "faces": faces
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<name>', methods=['DELETE'])
def delete_user(name):
    """Deletes a registered user's profile, embeddings, and sample images."""
    if engine is None:
        return jsonify({"status": "error", "message": "FaceEngine not initialized"}), 500
        
    try:
        success = engine.delete_user(name)
        if success:
            return jsonify({"status": "success", "message": f"User {name} deleted successfully."})
        else:
            return jsonify({"status": "error", "message": f"User {name} not found or failed to delete."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Start development server
if __name__ == '__main__':
    # Make sure required folders exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("registration_samples", exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
