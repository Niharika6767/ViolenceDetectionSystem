import numpy as np
# Monkey-patch np.sctypes for compatibility with imgaug in NumPy 2.0
if not hasattr(np, "sctypes"):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128]
    }

import sqlite3
import os
from flask import Flask, request, jsonify, session
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import imgaug.augmenters as iaa
import io
import base64
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry backend (no GUI)
import matplotlib.pyplot as plt
import queue
import threading
import smtplib
from email.mime.text import MIMEText
import time
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import hashlib
import secrets
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage
from flask_session import Session
import logging





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs


DB_PATH = "violence.db"  # Database file name

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
CORS(app, 
     supports_credentials=True,
     resources={r"/*": {
         "origins": "http://localhost:3000",
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"]
     }}
)
socketio = SocketIO(app, cors_allowed_origins="*")

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
Session(app)

# Live Streaming Variables
frame_queue = queue.Queue(maxsize=5)
violence_counter = 0
# For live stream alerts, we set a lower threshold
threshold_frames = 2



  # Change from 5 to 3 to trigger email alerts sooner
global stop_threads
stop_threads = False








# Load Pre-trained Model
model_path = r"D:\Violence-Alert-System\Violence Detection\modelnew.h5"
model = load_model(model_path)


# Global Storage for Analysis Results
results_log = []
alerts_log = []  # Stores detection results
daily_counts = {}  # Stores daily violence counts
daily_nonviolence_counts = {}  # Stores daily non-violence counts

# ---------------------- Helper Function: Extract Frames ----------------------
def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if not success:
            break
        if count % 7 == 0:
            resized_frame = cv2.resize(image, (128, 128))
            frames.append(resized_frame.astype('float32') / 255.0)
        count += 1
    vidcap.release()
    return np.array(frames)

IMG_SIZE = 128  # Set the image size

def video_to_frames(video):
    """
    Extracts frames from the video file.
    Processes every 7th frame, applies augmentations, converts BGR to RGB,
    and resizes the frame to (IMG_SIZE x IMG_SIZE).
    """
    vidcap = cv2.VideoCapture(video)
    count = 0
    ImageFrames = []
    while vidcap.isOpened():
        ID = vidcap.get(1)
        success, image = vidcap.read()
        if success:
            if (ID % 7 == 0):
                flip = iaa.Fliplr(1.0)
                zoom = iaa.Affine(scale=1.3)
                random_brightness = iaa.Multiply((1, 1.3))
                rotate = iaa.Affine(rotate=(-25, 25))

                image_aug = flip(image=image)
                image_aug = random_brightness(image=image_aug)
                image_aug = zoom(image=image_aug)
                image_aug = rotate(image=image_aug)

                rgb_img = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
                ImageFrames.append(resized)
            count += 1
        else:
            break
    vidcap.release()
    return ImageFrames

# Email configuration class
class EmailConfig:
    def __init__(self):
        self.current_email = None
        logger.debug("EmailConfig initialized")

email_config = EmailConfig()

@app.route('/api/set_email', methods=['POST'])
def set_email():
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
            
        # Set email in session
        session['email'] = email
        session.permanent = True
        
        return jsonify({'message': 'Email set successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def send_email_alert(violence_timestamp, confidence_score, video_path, frame_path):
    """Optimized email alert sending function."""
    if not email_config.current_email:
        print("⚠ No email configured for alerts")
        return

    try:
        # Start a new thread for sending email to avoid blocking
        def send_email_thread():
            try:
                msg = MIMEMultipart()
                msg['From'] = "niharikagoud45@gmail.com"
                msg['To'] = email_config.current_email
                msg['Subject'] = "⚠ Violence Detected Alert"

                # Email body
                body = f"""
                ⚠ Violence Detected!
                
                Time: {violence_timestamp}
                Confidence Score: {confidence_score:.2f}%
                
                Video has been saved for review.
                """
                msg.attach(MIMEText(body, 'plain'))

                # Attach the frame image
                if os.path.exists(frame_path):
                    with open(frame_path, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(frame_path))
                        msg.attach(img)

                # Connect to SMTP server and send email
                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    server.starttls()
                    server.login("niharikagoud45@gmail.com", "pbwr ktdl anwd bupz")
                    server.send_message(msg)
                    
                print("✅ Alert email sent successfully")
            except Exception as e:
                print(f"❌ Error sending email alert: {str(e)}")

        # Start the email thread
        threading.Thread(target=send_email_thread).start()
        
    except Exception as e:
        print(f"❌ Error preparing email alert: {str(e)}")

# ---------------------- Function to Save Data to SQLite ----------------------
def save_to_db(file_path, file_name, upload_time, detection_time, processing_time, status):
    """Save video detection details into SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results_data(file_path, file_name, upload_time, detection_time, processing_time, status)
            VALUES (?,?,?,?,?,?)
        ''', (file_path, file_name, upload_time, detection_time, processing_time, status))
        conn.commit()

# ---------------------- Endpoint: Video Analysis (Uploaded Videos) ----------------------
@app.route('/api/detect', methods=['POST'])
def detect():
    global alerts_log, daily_counts, daily_nonviolence_counts

    print("📌 detect() function called!")  # Debugging

    if 'video' not in request.files:
        print("❌ No video file provided!")
        return jsonify({'error': 'No video file provided.'}), 400

    video_file = request.files['video']
    file_name = video_file.filename  # Get file name
    file_path = os.path.abspath(file_name)
    temp_video_path = 'temp_video.mp4'
    video_file.save(temp_video_path)

    print("📌 Video saved successfully:", temp_video_path)

    start_time = datetime.now()
    frames = video_to_frames(temp_video_path)

    if not frames:
        os.remove(temp_video_path)
        print("❌ No frames extracted from video!")
        return jsonify({'error': 'No frames extracted from video.'}), 400

    print(f"📌 Extracted {len(frames)} frames")

    processed_frames = [frame.astype('float32') / 255.0 for frame in frames]
    processed_frames = np.array(processed_frames)

    predictions = model.predict(processed_frames)
    preds = predictions > 0.5
    n_violence = int(np.sum(preds))
    n_total = processed_frames.shape[0]

    os.remove(temp_video_path)

    end_time = datetime.now()
    processing_duration = (end_time - start_time).total_seconds()
    detection_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    today = end_time.strftime("%Y-%m-%d")
    detection_label = "Violence Detected" if n_violence > (n_total - n_violence) else "No Violence Detected"

    print(f"📌 {detection_label}: {n_violence} violent frames out of {n_total} frames.")

    # Save results to SQLite
    save_to_db(file_path, file_name, start_time.strftime("%Y-%m-%d %H:%M:%S"), detection_time, processing_duration, detection_label)

    alert = {
        'upload_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'detection_time': detection_time,
        'processing_duration': processing_duration,
        'message': f"{detection_label}: {n_violence} violent frames out of {n_total} frames.",
        'violence_frames': n_violence,
        'nonviolence_frames': n_total - n_violence
    }

    alerts_log.append(alert)

    if detection_label == "Violence Detected":
        daily_counts[today] = daily_counts.get(today, 0) + 1
    else:
        daily_nonviolence_counts[today] = daily_nonviolence_counts.get(today, 0) + 1

    print(f"🟢 Updated Counts - Violent: {daily_counts}, Non-Violent: {daily_nonviolence_counts}")

    return jsonify(alert)

# ---------------------- Endpoint: Dashboard Data ----------------------
@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    try:
        # Get the week offset from query parameters
        week_offset = int(request.args.get('week_offset', 0))
        
        with sqlite3.connect("violence.db") as conn:
            cursor = conn.cursor()

            # Get total counts
            cursor.execute("SELECT COUNT(*) FROM results_data")
            total_videos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM results_data WHERE status = 'Violence Detected'")
            violent_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM results_data WHERE status = 'No Violence Detected'")
            nonviolent_count = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(processing_time) FROM results_data")
            avg_processing_time = cursor.fetchone()[0] or 0

            # Get weekly data based on the offset
            days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            weekly_data = {day: {"violent": 0, "nonviolent": 0} for day in days_of_week}

            # Get the current date and calculate week boundaries
            cursor.execute("SELECT date('now', 'weekday 0', '-6 days')")  # Get Sunday of current week
            current_week_end = cursor.fetchone()[0]
            
            # Calculate the start and end dates for the requested week
            if week_offset == 0:
                # Current week (Monday to Sunday)
                cursor.execute("""
                    SELECT 
                        strftime('%w', upload_time) AS weekday,
                        COUNT(*) as count,
                        status,
                        date(upload_time) as upload_date,
                        date('now', 'weekday 0', '-6 days') as week_start,
                        date('now', 'weekday 0') as week_end
                    FROM results_data 
                    WHERE date(upload_time) >= date('now', 'weekday 0', '-6 days')  -- Monday
                    AND date(upload_time) <= date('now', 'weekday 0')  -- Sunday
                    GROUP BY weekday, status, upload_date
                    ORDER BY upload_date
                """)
            else:
                # Previous weeks (Monday to Sunday)
                cursor.execute(f"""
                    SELECT 
                        strftime('%w', upload_time) AS weekday,
                        COUNT(*) as count,
                        status,
                        date(upload_time) as upload_date,
                        date('now', 'weekday 0', '{(week_offset * 7) - 6} days') as week_start,
                        date('now', 'weekday 0', '{week_offset * 7} days') as week_end
                    FROM results_data 
                    WHERE date(upload_time) >= date('now', 'weekday 0', '{(week_offset * 7) - 6} days')  -- Monday of the week
                    AND date(upload_time) <= date('now', 'weekday 0', '{week_offset * 7} days')  -- Sunday of the week
                    GROUP BY weekday, status, upload_date
                    ORDER BY upload_date
                """)

            rows = cursor.fetchall()
            
            # Get the week dates for debugging
            if rows:
                week_start = rows[0][4]  # week_start from the query
                week_end = rows[0][5]    # week_end from the query
                print(f"🟢 Processing data for week: {week_start} to {week_end}")
            
            print(f"🟢 Query results for week offset {week_offset}:", rows)

            for row in rows:
                weekday = int(row[0])  # 0 = Sunday, 1 = Monday, ..., 6 = Saturday
                count = row[1]
                status = row[2]
                date = row[3]
                
                # Convert to our day format (0 = Monday, ..., 6 = Sunday)
                day_index = (weekday - 1) % 7
                day_name = days_of_week[day_index]
                
                print(f"🟢 Processing: date={date}, day={day_name}, count={count}, status={status}")
                
                if status == "Violence Detected":
                    weekly_data[day_name]["violent"] += count
                else:
                    weekly_data[day_name]["nonviolent"] += count

            print(f"🟢 Final weekly data:", weekly_data)
            return jsonify({
                "total_videos": total_videos,
                "violent_count": violent_count,
                "nonviolent_count": nonviolent_count,
                "avg_processing_time": round(avg_processing_time, 2),
                "weekly_data": weekly_data,
                "week_range": {
                    "start": week_start if rows else None,
                    "end": week_end if rows else None
                }
            })

    except Exception as e:
        print(f"❌ Error in /api/dashboard_data:", str(e))
        return jsonify({"error": "Failed to fetch dashboard data"}), 500

# ---------------------- Endpoint: Fetch Video Detection Records ----------------------
@app.route('/api/results', methods=['GET'])
def get_results():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_name, upload_time, detection_time, status FROM results_data")
        results = cursor.fetchall()

    results_list = [
        {"id": row[0], "file_name": row[1], "upload_time": row[2], "detection_time": row[3], "status": row[4]}
        for row in results
    ]
    return jsonify(results_list)

# ---------------------- Live Stream Processing ----------------------
violence_history = []  # Stores timestamps & frame counts
stop_threads = False
segment_duration = 1  # Reduced to 1 second for faster response
video_fps = 30
frames_per_segment = segment_duration * video_fps
violent_threshold = 0.80  # Slightly increased for better accuracy
violence_ratio_threshold = 0.4  # Lowered for faster detection
min_brightness_threshold = 10  # Lowered to handle more lighting conditions
confidence_buffer_size = 8  # Reduced for faster response
alert_cooldown = 60  # Reduced to 1 minute for more frequent alerts
required_consecutive_frames = 2  # Reduced for faster detection

def detect_violence_from_stream(video_url):
    """Continuously process the live stream with optimized violence detection."""
    if not email_config.current_email:
        print("⚠ No email configured for alerts in detection thread")
        return

    print(f"✅ Starting detection with email configured: {email_config.current_email}")
    
    # Generate a unique session ID for this stream
    session_id = f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("❌ Unable to open live stream. Check the IP address.")
        return

    print(f"✅ Live stream started successfully. Session ID: {session_id}")
    
    confidence_buffer = deque(maxlen=confidence_buffer_size)
    frame_buffer = []
    last_alert_time = 0
    consecutive_violent_frames = 0
    motion_history = deque(maxlen=8)  # Reduced for faster response

    while not stop_threads:
        violent_frames = 0
        total_frames = 0
        confidence_scores = []
        frames = []
        start_time = time.time()
        prev_frame = None

        while time.time() - start_time < segment_duration:
            ret, frame = cap.read()
            if not ret:
                print("❌ Stream ended or interrupted.")
                cap.release()
                return

            total_frames += 1
            
            # Enhanced motion detection
            if prev_frame is not None:
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(gray_prev, gray_curr)
                motion_score = np.mean(frame_diff)
                motion_history.append(motion_score)
            
            prev_frame = frame.copy()
            
            # Enhanced brightness check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness < min_brightness_threshold:
                # Skip frame if too dark
                continue
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (128, 128)).astype("float32") / 255.0
            frame_resized = frame_resized.reshape(1, 128, 128, 3)
            
            # Get prediction with multiple samples
            predictions = []
            for _ in range(2):  # Reduced to 2 predictions for faster processing
                pred = model.predict(frame_resized, verbose=0)[0]
                predictions.append(float(pred[0]))
            
            confidence = round(np.mean(predictions) * 100, 2)
            confidence_buffer.append(confidence)
            frames.append(frame)
            
            # Enhanced violence detection logic
            if len(confidence_buffer) >= 2:  # Reduced to 2 frames for faster response
                recent_confidence = list(confidence_buffer)[-2:]  # Check last 2 frames
                avg_recent_confidence = np.mean(recent_confidence)
                
                # Check motion level
                avg_motion = np.mean(motion_history) if motion_history else 0
                
                # Only consider frame as violent if there's significant motion
                if avg_motion > 5 and avg_recent_confidence > violent_threshold * 100:
                    consecutive_violent_frames += 1
                    violent_frames += 1
                else:
                    consecutive_violent_frames = 0
                
                confidence_scores.append(avg_recent_confidence)

        # Calculate violence metrics
        violence_ratio = violent_frames / total_frames if total_frames > 0 else 0
        avg_conf = np.mean(confidence_scores) if confidence_scores else 0
        
        print(f"🔍 Violence Ratio: {violence_ratio:.2f}, Avg Confidence: {avg_conf:.2f}%, Consecutive Frames: {consecutive_violent_frames}, Motion: {np.mean(motion_history) if motion_history else 0:.2f}")

        # Enhanced alert triggering logic
        current_time = time.time()
        should_alert = (
            (violence_ratio >= violence_ratio_threshold or consecutive_violent_frames >= required_consecutive_frames) and
            avg_conf > violent_threshold * 100 and
            current_time - last_alert_time >= alert_cooldown and
            (np.mean(motion_history) if motion_history else 0) > 5
        )

        if should_alert:
            alert_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            print(f"🚨 Violence detected at {alert_time} with confidence {avg_conf:.2f}% and motion {np.mean(motion_history):.2f}")
            
            # Create directory if it doesn't exist
            os.makedirs("violent_videos_logger", exist_ok=True)
            
            # Save video segment with timestamp
            save_path = f"violent_videos_logger/violent_{alert_time}.mp4"
            save_video_segment(frames, save_path)
            
            # Save frame with timestamp
            frame_path = f"violent_videos_logger/frame_{alert_time}.jpg"
            cv2.imwrite(frame_path, frames[-1])
            
            # Store analytics data
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO livestream_analytics 
                    (session_id, violence_detected, confidence_score, violence_ratio, motion_level, clip_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    True,
                    avg_conf,
                    violence_ratio,
                    float(np.mean(motion_history) if motion_history else 0),
                    save_path
                ))
                conn.commit()

            # Send immediate alert with both video and frame
            send_email_alert(alert_time, avg_conf, save_path, frame_path)
            
            # Emit real-time alert via SocketIO
            socketio.emit('detection_result', {
                'violence_detected': 1,
                'alert_time': alert_time,
                'confidence': avg_conf,
                'violence_ratio': violence_ratio,
                'motion_level': np.mean(motion_history)
            })
            
            last_alert_time = current_time

def save_video_segment(frames, save_path):
    """Saves detected violent frames as a video with proper FPS."""
    if not frames:
        print("No frames to save!")
        return
        
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, video_fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"📁 Video saved: {save_path}")


# ---------------------- Live Stream Endpoints ----------------------
@app.route('/start_live_stream', methods=['POST'])
def start_live_stream():
    """Start live streaming from IP Webcam."""
    logger.debug(f"Session data: {dict(session)}")
    logger.debug(f"Email config: {email_config.current_email}")
    
    data = request.get_json()
    video_url = data.get('ip')
    if not video_url:
        return jsonify({'error': 'No IP provided'}), 400

    # Get the current user's email from session
    if not email_config.current_email:
        if 'email' in session:
            email_config.current_email = session['email']
            logger.debug(f"Email configured from session: {email_config.current_email}")
        else:
            logger.debug("No email in session")
            return jsonify({'error': 'No email configured for alerts. Please log in again.'}), 400

    global stop_threads
    stop_threads = False
    threading.Thread(target=detect_violence_from_stream, args=(video_url,)).start()
    return jsonify({'message': 'Live stream started successfully'}), 200

@app.route('/stop_live_stream', methods=['POST'])
def stop_live_stream():
    """Stop live streaming."""
    # global stop_threads
    stop_threads = True
    return jsonify({'message': 'Live stream stopped successfully'}), 200

@socketio.on('connect')
def handle_connect():
    print("🔗 Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ Client disconnected")

# ---------------------- Save Analysis Data ----------------------
@app.route('/save_analysis', methods=['POST'])
def save_analysis():
    data = request.json['records']
    conn = sqlite3.connect('analysis.db')
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            violence_frames INTEGER,
            confidence REAL
        )
    """)

    for record in data:
        cursor.execute("INSERT INTO Analysis (timestamp, violence_frames, confidence) VALUES (?, ?, ?)",
                       (record['timestamp'], record['violenceFrames'], record['confidence']))

    conn.commit()
    conn.close()
    return jsonify({"message": "Analysis data saved!"}), 200

# Add new endpoint for livestream analytics
@app.route('/api/livestream_analytics', methods=['GET'])
def get_livestream_analytics():
    try:
        timeframe = request.args.get('timeframe', 'day')  # day, week, month
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            if timeframe == 'day':
                time_filter = "timestamp >= date('now', '-1 day')"
            elif timeframe == 'week':
                time_filter = "timestamp >= date('now', '-7 days')"
            else:  # month
                time_filter = "timestamp >= date('now', '-30 days')"
            
            # Get hourly distribution
            cursor.execute(f"""
                SELECT 
                    strftime('%H', datetime(timestamp, 'localtime')) as hour,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(violence_ratio) as avg_violence_ratio,
                    AVG(motion_level) as avg_motion
                FROM livestream_analytics
                WHERE {time_filter}
                GROUP BY hour
                ORDER BY hour
            """)
            
            hourly_data = cursor.fetchall()
            
            # Get session-based data
            cursor.execute(f"""
                SELECT 
                    session_id,
                    datetime(MIN(timestamp), 'localtime') as start_time,
                    datetime(MAX(timestamp), 'localtime') as end_time,
                    COUNT(*) as incident_count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(violence_ratio) as avg_violence_ratio,
                    AVG(motion_level) as avg_motion
                FROM livestream_analytics
                WHERE {time_filter}
                GROUP BY session_id
                ORDER BY start_time DESC
            """)
            
            session_data = cursor.fetchall()
            
            # Get total incidents
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM livestream_analytics 
                WHERE {time_filter}
            """)
            total_incidents = cursor.fetchone()[0]
            
            return jsonify({
                'total_incidents': total_incidents,
                'hourly_distribution': [{
                    'hour': row[0],
                    'count': row[1],
                    'avg_confidence': row[2],
                    'avg_violence_ratio': row[3],
                    'avg_motion': row[4]
                } for row in hourly_data],
                'sessions': [{
                    'session_id': row[0],
                    'start_time': row[1],
                    'end_time': row[2],
                    'incident_count': row[3],
                    'avg_confidence': row[4],
                    'avg_violence_ratio': row[5],
                    'avg_motion': row[6],
                    'duration': round((datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S') - 
                                    datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')).total_seconds() / 60)
                } for row in session_data]
            })
            
    except Exception as e:
        print(f"❌ Error in /api/livestream_analytics:", str(e))
        return jsonify({"error": "Failed to fetch livestream analytics"}), 500

# Initialize database
def init_db():
    """Initialize the database with required tables."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if password column exists
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'password_hash' not in columns:
                # Add password column if it doesn't exist
                cursor.execute('ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ""')
                print("Added password_hash column to users table")
        else:
            # Create users table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            print("Created users table")
        
        conn.commit()

# Call init_db when the application starts
init_db()

# Add session secret key
app.secret_key = secrets.token_hex(16)

@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        print("Registration attempt received")  # Debug log
        data = request.json
        print(f"Received data: {data}")  # Debug log
        
        if not data:
            print("No data provided")  # Debug log
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        password = data.get('password')
        
        print(f"Email: {email}, Password length: {len(password) if password else 0}")  # Debug log
        
        if not email or not password:
            print("Missing email or password")  # Debug log
            return jsonify({'error': 'Email and password are required'}), 400
            
        # Validate email format
        if '@' not in email or '.' not in email:
            print("Invalid email format")  # Debug log
            return jsonify({'error': 'Invalid email format'}), 400
            
        # Validate password length
        if len(password) < 6:
            print("Password too short")  # Debug log
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # First, check if the email exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                print("Email already registered")  # Debug log
                return jsonify({'error': 'Email already registered'}), 400
            
            # If email doesn't exist, insert new user
            cursor.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', 
                         (email, hashed_password))
            conn.commit()
            
            # Verify the insertion
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if not cursor.fetchone():
                print("Failed to verify user creation")  # Debug log
                return jsonify({'error': 'Failed to create user'}), 500
                
        print("Registration successful")  # Debug log
        return jsonify({'message': 'Registration successful'}), 201
    except sqlite3.IntegrityError as e:
        print(f"Database integrity error: {str(e)}")  # Debug log
        return jsonify({'error': 'Email already registered'}), 400
    except Exception as e:
        print(f"Registration error: {str(e)}")  # Debug log
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
            
        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, email FROM users WHERE email = ? AND password_hash = ?',
                         (email, hashed_password))
            user = cursor.fetchone()
            
            if user:
                # Set session data
                session.clear()  # Clear any existing session
                session['user_id'] = user[0]
                session['email'] = email
                session.permanent = True
                
                # Configure email for alerts
                email_config.current_email = email
                logger.debug(f"Login successful. Session: {dict(session)}, Email config: {email_config.current_email}")
                
                return jsonify({
                    'message': 'Login successful',
                    'email': email,
                    'user_id': user[0]
                }), 200
            else:
                logger.debug("Invalid credentials")
                return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    # Clear email configuration and session when logging out
    email_config.current_email = None
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

# ---------------------- Run Flask App ----------------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

def check_and_fix_users_table():
    """Check and fix the users table if needed."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("Users table does not exist, creating it...")
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            return
        
        # Check table structure
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        required_columns = ['id', 'email', 'password_hash', 'created_at']
        
        # If any required column is missing, recreate the table
        if not all(col in columns for col in required_columns):
            print("Users table has incorrect structure, recreating it...")
            # Backup existing data
            cursor.execute("SELECT * FROM users")
            old_data = cursor.fetchall()
            
            # Drop and recreate table
            cursor.execute("DROP TABLE IF EXISTS users")
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Reinsert data if possible
            if old_data:
                for row in old_data:
                    try:
                        cursor.execute('''
                            INSERT INTO users (email, password_hash, created_at)
                            VALUES (?, ?, ?)
                        ''', (row[1], row[2], row[3]))
                    except:
                        continue
            
            conn.commit()

# Call this function at startup
check_and_fix_users_table()