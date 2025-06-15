import os
from flask import Flask, render_template, Response, request, redirect, url_for, make_response
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
from database import (
    save_face_encoding,
    get_all_face_encodings,
    mark_attendance_db,
    get_attendance_records,
    get_user_image
)

app = Flask(__name__)

# Production settings
app.config['PRODUCTION'] = os.environ.get('PRODUCTION', False)

# Initialize camera only in development
camera = None
if not app.config['PRODUCTION']:
    try:
        camera = cv2.VideoCapture(0)
    except Exception as e:
        print(f"Camera initialization error: {e}")

# Ensure data directories exist
os.makedirs('known_faces', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Ensure Excel file exists
if not os.path.exists('attendance.xlsx'):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel('attendance.xlsx', index=False)

def create_message_frame(message="Camera not available in production mode"):
    # Create a black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add text to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)  # White text
    thickness = 2
    
    # Get the text size
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    
    # Calculate text position to center it
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    
    # Put the text on the frame
    cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

def gen_frames():
    # Check if in production mode
    if app.config['PRODUCTION']:
        while True:
            frame = create_message_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    # Define a threshold for face recognition - lower is more strict
    FACE_RECOGNITION_THRESHOLD = 0.5
    
    while True:
        success, frame = camera.read()
        if not success:
            frame = create_message_frame("Camera error - trying to reconnect...")
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue
            
        # Get latest face encodings for each frame
        known_encodings, known_names = get_all_face_encodings()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0

            if len(known_encodings) > 0:
                # Ensure face_encoding is the correct shape
                face_encoding = face_encoding.reshape(128,)
                
                try:
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    matches = face_distances <= FACE_RECOGNITION_THRESHOLD
                    
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            mark_attendance_db(name)
                except Exception as e:
                    print(f"Error in face comparison: {e}")
                    continue

            # Draw name and confidence score
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            if name != "Unknown":
                text = f"{name} ({confidence:.2%})"
            else:
                text = name
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        success, frame = camera.read()
        if success:
            # Convert frame to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                return render_template('register.html', error="No face detected! Please try again.")
            
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            
            # Load existing faces and check for matches
            known_encodings, known_names = get_all_face_encodings()
            
            # Check if there are any existing face encodings
            if len(known_encodings) > 0:
                try:
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    # Use a strict threshold for registration
                    REGISTRATION_THRESHOLD = 0.4
                    if min_distance <= REGISTRATION_THRESHOLD:
                        existing_name = known_names[best_match_index]
                        confidence = (1 - min_distance) * 100
                        return render_template('register.html', 
                                            error=f"This face is already registered under the name: {existing_name} (Match confidence: {confidence:.1f}%)")
                except Exception as e:
                    print(f"Error during face comparison: {e}")
                    return render_template('register.html', 
                                        error="An error occurred during face comparison. Please try again.")
            
            # If no match found, proceed with registration
            save_face_encoding(name, face_encoding, frame)
            return redirect(url_for('index'))
            
    return render_template('register.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    records = get_attendance_records()
    return render_template('attendance.html', records=records)

@app.route('/user_image/<name>')
def user_image(name):
    image = get_user_image(name)
    if image is not None:
        _, buffer = cv2.imencode('.jpg', image)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        return response
    return "Image not found", 404

def get_dummy_frame():
    """Create a dummy frame for production environment"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera disabled in production", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

if __name__ == '__main__':
    app.run(debug=True)
