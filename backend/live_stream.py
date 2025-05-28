

import numpy as np

# Monkey-patch np.sctypes for compatibility with imgaug in NumPy 2.0
if not hasattr(np, "sctypes"):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128]
    }
import time
import threading
from flask import Flask, request, jsonify, Response
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Load pre-trained model (adjust model_path if needed)
model_path = r"D:\Violence-Alert-System\Violence Detection\modelnew.h5"
model = load_model(model_path)

# Global variables for live stream processing
last_detection = "No Violence Detected"
live_stream_url = None
streaming = False
analysis_data = []
frame_skip = 5  # Process every 5th frame for performance
lock = threading.Lock()

def preprocess_frame(frame):
    """Resize and normalize frame for prediction."""
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def detect_violence(frame):
    """Return True if violence is detected (probability > 0.5)."""
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)[0]
    return prediction > 0.5

def analyze_live_stream():
    """Continuously process the live stream and perform violence detection."""
    global streaming, live_stream_url, analysis_data, last_detection
    cap = cv2.VideoCapture(live_stream_url)
    if not cap.isOpened():
        print("❌ Failed to open live stream at", live_stream_url)
        streaming = False
        socketio.emit("live_error", {"error": "Failed to open live stream"})
        return

    print("✅ Live stream started.")
    start_time = None
    violent_frame_count = 0
    frame_count = 0
    while streaming:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Error reading frame; exiting analysis loop.")
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        is_violent = detect_violence(frame)
        with lock:
            last_detection = "Violence Detected" if is_violent else "No Violence Detected"
        # Emit the detection status to the frontend
        socketio.emit("live_analysis", {"violence": is_violent})
        if is_violent:
            if start_time is None:
                start_time = time.strftime("%H:%M:%S")
            violent_frame_count += 1
        else:
            if start_time:
                end_time = time.strftime("%H:%M:%S")
                analysis_data.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "violent_frames": violent_frame_count
                })
                start_time = None
                violent_frame_count = 0
    cap.release()
    print("⏹️ Live stream stopped.")

def generate_frames(ip):
    """Generator that yields JPEG frames from the IP webcam stream."""
    stream_url = f"http://{ip}/video"
    print("Opening stream URL:", stream_url)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ Could not open stream at", stream_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No frame received; breaking loop.")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
    print("Stream closed for", stream_url)

@app.route('/video_feed')
def video_feed():
    """Endpoint to return live stream frames as a multipart response."""
    ip = request.args.get("ip")
    return Response(generate_frames(ip), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_live_stream", methods=["POST"])
def start_live_stream():
    """Start live stream analysis based on IP provided by the frontend."""
    global streaming, live_stream_url
    data = request.get_json()
    ip = data.get("ip")
    if not ip:
        return jsonify({"error": "No IP provided"}), 400
    # Construct the live stream URL (e.g. "http://192.168.1.100:8080/video")
    live_stream_url = f"http://{ip}/video"
    streaming = True
    threading.Thread(target=analyze_live_stream, daemon=True).start()
    return jsonify({"message": "Live stream started"}), 200

@app.route("/stop_live_stream", methods=["POST"])
def stop_live_stream():
    """Stop live stream analysis."""
    global streaming
    streaming = False
    return jsonify({"message": "Live stream stopped"}), 200

@app.route("/api/analysis_data", methods=["GET"])
def get_analysis_data():
    """Return the analysis data collected from the live stream."""
    return jsonify(analysis_data)

@socketio.on("connect")
def handle_connect():
    print("🔗 Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("❌ Client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)