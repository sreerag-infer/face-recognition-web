import os
import logging
import base64
import time

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from face_engine import FaceEngine, sanitize_name

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
CORS(app)

# ---------------------------------------------------------------------------
# Initialize FaceEngine (singleton — loaded once at startup)
# ---------------------------------------------------------------------------
engine = None

def _init_engine():
    global engine
    try:
        engine = FaceEngine()
        logger.info("FaceEngine initialized successfully.")
    except Exception:
        logger.exception("CRITICAL: Failed to initialize FaceEngine.")
        engine = None

_init_engine()

# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _require_engine():
    """Returns a JSON error response if the engine is not available, else None."""
    if engine is None:
        return jsonify({"status": "error", "message": "FaceEngine is not initialized. Check server logs."}), 503
    return None


def _get_json_body():
    """Safely parse JSON request body."""
    data = request.get_json(silent=True)
    if data is None:
        return {}
    return data


def decode_base64_image(base64_str):
    """Decodes a base64 image string (data URI or raw) into a CV2 BGR image."""
    try:
        if not base64_str or not isinstance(base64_str, str):
            return None
        # Strip data URI prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        logger.debug("Failed to decode base64 image.", exc_info=True)
        return None


def encode_image_to_base64(img):
    """Encodes a CV2 BGR image into a base64 JPEG data URI."""
    try:
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        logger.debug("Failed to encode image to base64.", exc_info=True)
        return ""


# ===========================================================================
# Page routes
# ===========================================================================

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

# ===========================================================================
# Health & status endpoints
# ===========================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Lightweight health probe for load balancers and monitoring."""
    ok = engine is not None
    return jsonify({
        "healthy": ok,
        "engine": "loaded" if ok else "unavailable",
    }), 200 if ok else 503


@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns backend system status and statistics."""
    err = _require_engine()
    if err:
        return err

    providers = []
    try:
        if hasattr(engine.model, 'providers'):
            providers = list(engine.model.providers)
    except Exception:
        providers = ["CPUExecutionProvider"]

    return jsonify({
        "status": "ok",
        "message": "System is running",
        "users_count": len(engine.database),
        "providers": providers,
    })


# ===========================================================================
# User management
# ===========================================================================

@app.route('/api/people', methods=['GET'])
@app.route('/api/users', methods=['GET'])
def get_users():
    """Returns list of registered users."""
    err = _require_engine()
    if err:
        return err

    try:
        users = engine.get_registered_users()
        return jsonify({"users": users})
    except Exception:
        logger.exception("Error fetching user list.")
        return jsonify({"users": [], "error": "Internal server error."}), 500


@app.route('/api/users/<name>', methods=['DELETE'])
def delete_user(name):
    """Deletes a registered user's profile, embeddings, and sample images."""
    err = _require_engine()
    if err:
        return err

    if not name or not name.strip():
        return jsonify({"status": "error", "message": "User name is required."}), 400

    try:
        clean_name = sanitize_name(name)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    try:
        success = engine.delete_user(clean_name)
        if success:
            return jsonify({"status": "success", "message": f"User '{clean_name}' deleted successfully."})
        else:
            return jsonify({"status": "error", "message": f"User '{clean_name}' not found."}), 404
    except Exception:
        logger.exception("Error deleting user '%s'.", clean_name)
        return jsonify({"status": "error", "message": "Internal server error."}), 500


# ===========================================================================
# Registration workflow
# ===========================================================================

@app.route('/api/register/start', methods=['POST'])
def register_start():
    """Starts a registration session."""
    err = _require_engine()
    if err:
        return err

    data = _get_json_body()
    name = data.get('name', '').strip()

    if not name:
        return jsonify({"status": "error", "message": "Person name is required."}), 400

    try:
        clean_name = sanitize_name(name)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    already_exists = clean_name in engine.database

    try:
        session_name = engine.start_registration(name)
        return jsonify({
            "status": "success",
            "name": session_name,
            "already_exists": already_exists,
            "message": f"Registration session started for {session_name}."
        })
    except Exception:
        logger.exception("Error starting registration for '%s'.", name)
        return jsonify({"status": "error", "message": "Internal server error."}), 500


@app.route('/api/register/frame', methods=['POST'])
def register_frame():
    """Processes a single frame for registration."""
    err = _require_engine()
    if err:
        return err

    data = _get_json_body()
    name = data.get('name', '').strip()
    image_base64 = data.get('image', '')

    if not name:
        return jsonify({"success": False, "message": "Person name is required."}), 400

    if not image_base64:
        return jsonify({"success": False, "message": "Image frame is required."}), 400

    img = decode_base64_image(image_base64)
    if img is None:
        return jsonify({"success": False, "message": "Invalid or corrupt image data."}), 400

    try:
        success, message, count = engine.process_registration_frame(name, img, time.time())
        return jsonify({
            "success": success,
            "message": message,
            "count": count
        })
    except Exception:
        logger.exception("Error processing registration frame for '%s'.", name)
        return jsonify({"success": False, "message": "Internal server error."}), 500


@app.route('/api/register/finish', methods=['POST'])
def register_finish():
    """Completes the registration process, saving npy files."""
    err = _require_engine()
    if err:
        return err

    data = _get_json_body()
    name = data.get('name', '').strip()

    if not name:
        return jsonify({"status": "error", "message": "Person name is required."}), 400

    try:
        success, message = engine.finish_registration(name)
        if success:
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "message": message}), 400
    except Exception:
        logger.exception("Error finishing registration for '%s'.", name)
        return jsonify({"status": "error", "message": "Internal server error."}), 500


@app.route('/api/register/cancel', methods=['POST'])
def register_cancel():
    """Cancels registration and cleans up."""
    err = _require_engine()
    if err:
        return err

    data = _get_json_body()
    name = data.get('name', '').strip()

    if not name:
        return jsonify({"status": "error", "message": "Person name is required."}), 400

    try:
        engine.cancel_registration(name)
        return jsonify({"status": "success", "message": "Registration cancelled."})
    except Exception:
        logger.exception("Error cancelling registration for '%s'.", name)
        return jsonify({"status": "error", "message": "Internal server error."}), 500


# ===========================================================================
# Recognition
# ===========================================================================

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Runs face recognition on the uploaded frame."""
    err = _require_engine()
    if err:
        return err

    data = _get_json_body()
    image_base64 = data.get('image', '')

    if not image_base64:
        return jsonify({"error": "No image frame provided."}), 400

    img = decode_base64_image(image_base64)
    if img is None:
        return jsonify({"error": "Invalid or corrupt image data."}), 400

    try:
        annotated_img, faces = engine.recognize_faces(img)
        annotated_base64 = encode_image_to_base64(annotated_img)
        return jsonify({
            "image": annotated_base64,
            "faces": faces
        })
    except Exception:
        logger.exception("Error during face recognition.")
        return jsonify({"error": "Internal server error."}), 500


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("registration_samples", exist_ok=True)

    port = int(os.environ.get("PORT", 5000))
    debug = False

    logger.info("Starting Flask development server on port %d (debug=%s).", port, debug)
    app.run(host='0.0.0.0', port=port, debug=debug)
