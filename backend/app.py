from flask import Flask, render_template, Response, request, redirect, url_for, make_response, make_response
import cv2
import face_recognition
import numpy as np
import os
import base64
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

# Ensure Excel file exists
if not os.path.exists('attendance.xlsx'):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel('attendance.xlsx', index=False)

camera = cv2.VideoCapture(0)

def gen_frames():
    # Define a threshold for face recognition - lower is more strict
    FACE_RECOGNITION_THRESHOLD = 0.5
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
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

if __name__ == '__main__':
    app.run(debug=True)
