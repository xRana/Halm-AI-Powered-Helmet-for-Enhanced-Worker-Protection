from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os
import random 

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for all routes to prevent browser blocking (allows frontend to communicate with backend)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- 1. Database Configuration ---
# Set up SQLite database for local development and testing
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///halm_local.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Initialize the database extension
db = SQLAlchemy(app)

# --- 2. Database Models ---

# User model: Represents system administrators or safety officers
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20))
    department = db.Column(db.String(100))

# Worker model: Stores information about the construction workers monitored by the system
class Worker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(100))       # Job title: Electrician, Welder, etc.
    age = db.Column(db.Integer)            # Worker's age
    blood_type = db.Column(db.String(10))  # Important for medical emergencies
    medical_info = db.Column(db.String(200)) # Any existing medical conditions

# Alert model: Logs safety hazards detected by the AI camera
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))
    type = db.Column(db.String(50))        # E.g., Sharp Tool, Pothole
    severity = db.Column(db.String(20))    # E.g., Critical, Warning
    status = db.Column(db.String(20))      # Pending or Resolved
    description = db.Column(db.String(200)) # Detailed description of the event
    worker_name = db.Column(db.String(100)) # Name of the worker near the hazard
    confidence = db.Column(db.Float)       # AI detection confidence score
    lat = db.Column(db.Float)              # GPS Latitude
    lng = db.Column(db.Float)              # GPS Longitude

# Create all database tables based on the models defined above
with app.app_context():
    db.create_all()

latest_worker_location = {"lat": 25.3463, "lng": 49.5937}

# --- 3. AI Setup ---
# Define the path to the custom YOLOv8 model weights
model_path = 'best.pt'
# Check if the custom model exists, otherwise load the default YOLOv8 nano model as a fallback
if not os.path.exists(model_path):
    print(f"⚠️ Warning: Model not found. Using fallback.")
    model = YOLO('yolov8n.pt') 
else:
    print(f"✅ Model loaded: {model_path}")
    model = YOLO(model_path)

# Variables to manage alert frequency and prevent database spam
last_alert_time = 0
ALERT_COOLDOWN = 10 # Wait 10 seconds before logging another alert
# Classes that the AI model is specifically trained to detect as hazards
HAZARD_CLASSES = ['sharp tool', 'sharp tool (hazard)', 'pothole', 'pothole (hazard)']

# --- Hardware Integration Helper ---
def get_hardware_location():
    global latest_worker_location
    return latest_worker_location["lat"], latest_worker_location["lng"]

def get_registered_worker_from_db():
    """
    Fetches a random worker name from the registered workers in the database.
    """
    with app.app_context():
        try:
            # Query a random worker from the Worker table
            worker = Worker.query.order_by(db.func.random()).first()
            if worker:
                return worker.name
        except Exception as e:
            print(f"Database Error: {e}")
    return "Guest Worker" # Fallback if no workers are found in DB

# --- 4. Camera & Detection Logic ---
def generate_frames():
    # Generator function to capture video frames, run AI inference, and stream to frontend
    global last_alert_time
    # Open default system camera (Index 0)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened(): 
        print("❌ Error: Camera not accessible.")
        return

    try:
        while True:
            # Read a frame from the camera feed
            success, frame = camera.read()
            if not success: break

            # Run YOLOv8 object detection on the current frame
            results = model(frame, conf=0.5, verbose=False)
            # Draw bounding boxes and labels on the frame
            annotated_frame = results[0].plot()
            current_time = time.time()
            
            # Iterate through detected objects
            for r in results:
                for box in r.boxes:
                    # Get class ID, name, and confidence score
                    cls_id = int(box.cls[0])
                    clean_name = model.names[cls_id].lower().strip()
                    conf_score = float(box.conf[0])

                    # Check if the detected object is considered a hazard
                    if clean_name in HAZARD_CLASSES:
                        # Check if cooldown period has passed to avoid spamming alerts
                        if (current_time - last_alert_time) > ALERT_COOLDOWN:
                            # Determine the specific hazard type
                            hazard_type = "Sharp Tool" if 'sharp' in clean_name else "Pothole"
                            
                            # Fetch dynamic worker and location data
                            detected_worker = get_registered_worker_from_db()
                            lat, lng = get_hardware_location()

                            # Create and save the new alert to the database
                            with app.app_context():
                                new_alert = Alert(
                                    date=datetime.now().strftime("%Y-%m-%d"),
                                    time=datetime.now().strftime("%I:%M:%S %p"),
                                    type=hazard_type, 
                                    severity='critical', 
                                    status='Pending',
                                    description=f'{clean_name} detected near {detected_worker}',
                                    worker_name=detected_worker,
                                    confidence=round(conf_score * 100, 1),
                                    lat=lat, 
                                    lng=lng
                                )
                                db.session.add(new_alert)
                                db.session.commit()
                            
                            # Log alert to the backend console
                            print(f"🚨 ALERT: {hazard_type} | Worker: {detected_worker} | Location: {lat}, {lng}")
                            last_alert_time = current_time

            # Encode the annotated frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            # Yield the frame in byte format for multipart HTTP streaming
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    except GeneratorExit:
        # Handle client disconnecting from the video stream gracefully
        print("Client disconnected.")
    finally:
        # Always release the camera resource when done to prevent memory leaks
        camera.release()

# --- 5. Routes ---

# Route: Video Feed Endpoint
@app.route('/video_feed')
def video_feed():
    # Returns the multipart HTTP response containing the MJPEG video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route: User Registration
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    # Check if email already exists in the database
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"success": False, "message": "Email already exists"}), 400
    
    # Create and save a new user (Safety Officer/Admin)
    new_user = User(
        username=data['username'], email=data['email'], password=data['password'],
        phone="", department="Safety Dept"
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"success": True}), 201

# Route: User Login
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    # Verify credentials against the database records
    user = User.query.filter_by(email=data['email'], password=data['password']).first()
    if user:
        # Return user details upon successful authentication
        return jsonify({
            "success": True, 
            "username": user.username, 
            "email": user.email, 
            "phone": user.phone, 
            "department": user.department
        }), 200
    # Return error if credentials don't match
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

# Route: Get User Information
@app.route('/api/user/info', methods=['POST'])
def get_user_info():
    data = request.json
    # Fetch user details by email
    user = User.query.filter_by(email=data['email']).first()
    if user:
        return jsonify({
            "username": user.username, "email": user.email, 
            "phone": user.phone, "department": user.department
        })
    return jsonify({"error": "User not found"}), 404

# Route: Update User Profile
@app.route('/api/user/update', methods=['POST'])
def update_user():
    data = request.json
    # Find user by their current email
    user = User.query.filter_by(email=data['current_email']).first()
    if user:
        # Update user fields with newly provided data
        user.username = data['username']
        user.email = data['email']
        user.phone = data['phone']
        user.department = data['department']
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"success": False}), 404

# Route: Fetch All Alerts
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    # Retrieve all alerts ordered by ID descending (newest first)
    alerts = Alert.query.order_by(Alert.id.desc()).all()
    # Serialize alert objects to JSON format
    return jsonify([{
        'id': a.id, 'date': a.date, 'time': a.time, 'type': a.type,
        'severity': a.severity, 'status': a.status, 'description': a.description,
        'worker_name': a.worker_name, 'confidence': a.confidence, 'lat': a.lat, 'lng': a.lng
    } for a in alerts])

# Route: Mark Alert as Resolved
@app.route('/api/alerts/resolve/<int:id>', methods=['POST'])
def resolve(id):
    # Find alert by ID and update its status
    alert = Alert.query.get(id)
    if alert:
        alert.status = 'Resolved'
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"success": False}), 404

# Route: Clear All Alerts
@app.route('/api/alerts/clear', methods=['POST'])
def clear():
    # Delete all records from the Alert table
    db.session.query(Alert).delete()
    db.session.commit()
    return jsonify({"success": True})

# Route: Fetch All Workers
@app.route('/api/workers', methods=['GET'])
def get_workers():
    # Retrieve all workers from the database
    workers = Worker.query.all()
    # Serialize worker objects to JSON format
    return jsonify([{
        'id': w.id, 'name': w.name, 'role': w.role, 
        'age': w.age, 'blood_type': w.blood_type, 'medical_info': w.medical_info
    } for w in workers])

# Route: Add a New Worker
@app.route('/api/workers', methods=['POST'])
def add_worker():
    data = request.json
    # Create and save a new worker record
    new_worker = Worker(
        name=data['name'], role=data['role'], age=data['age'],
        blood_type=data['blood_type'], medical_info=data['medical_info']
    )
    db.session.add(new_worker)
    db.session.commit()
    return jsonify({"success": True})

# Route: Delete a Worker
@app.route('/api/workers/<int:id>', methods=['DELETE'])
def delete_worker(id):
    # Find worker by ID and remove them from the database
    worker = Worker.query.get(id)
    if worker:
        db.session.delete(worker)
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"success": False}), 404

# Route: Update Worker Information (New)
@app.route('/api/workers/<int:id>', methods=['PUT'])
def update_worker_db(id):
    try:
        data = request.json
        # Find the specific worker by ID
        worker = Worker.query.get(id)
        
        if not worker:
            return jsonify({"success": False, "message": "Worker not found"}), 404

        # Update fields with new data coming from the frontend (using fallback to current value if not provided)
        worker.name = data.get('name', worker.name)
        worker.role = data.get('role', worker.role)
        worker.age = data.get('age', worker.age)
        worker.blood_type = data.get('blood_type', worker.blood_type)
        worker.medical_info = data.get('medical_info', worker.medical_info)

        # Save the changes to the database
        db.session.commit()
        return jsonify({"success": True, "message": "Worker updated successfully"}), 200
        
    except Exception as e:
        # Handle any unexpected server or database errors
        print(f"Update Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ==========================================
# Hardware Integration Routes
# ==========================================

# Endpoint 1: Receives GPS coordinates from the ESP32
@app.route('/api/hardware/gps', methods=['POST'])
def receive_hardware_gps():
    global latest_worker_location
    data = request.json
    
    if data and 'latitude' in data and 'longitude' in data:
        # Update the location with the latest reading from the ESP32
        latest_worker_location['lat'] = data['latitude']
        latest_worker_location['lng'] = data['longitude']
        print(f"🌍 GPS Updated: {latest_worker_location['lat']}, {latest_worker_location['lng']}")
        return jsonify({"success": True}), 200
        
    return jsonify({"success": False, "message": "Missing GPS data"}), 400

# Endpoint 2: Receives fall or hazard alerts from the Raspberry Pi
@app.route('/api/hardware/alert', methods=['POST'])
def receive_hardware_alert():
    global latest_worker_location
    data = request.json
    
    hazard_type = data.get('type', 'Unknown Hazard')
    confidence_score = data.get('confidence', 100.0)
    
    with app.app_context():
        detected_worker = get_registered_worker_from_db()
        
        # Create a new alert in the database and link it to the last recorded GPS location
        new_alert = Alert(
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%I:%M:%S %p"),
            type=hazard_type,
            severity='critical',
            status='Pending',
            description=f"{hazard_type} detected for worker: {detected_worker}",
            worker_name=detected_worker,
            confidence=confidence_score,
            lat=latest_worker_location['lat'],
            lng=latest_worker_location['lng']
        )
        db.session.add(new_alert)
        db.session.commit()
        
    print(f"🚨 Emergency from Raspberry Pi! Hazard: {hazard_type}")
    return jsonify({"success": True, "message": "Alert saved successfully"}), 201

# Entry point for the application
if __name__ == '__main__':
    # Run the Flask development server on localhost port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)