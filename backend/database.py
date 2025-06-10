from pymongo import MongoClient
from dotenv import load_dotenv
import os
import numpy as np
import pickle
import base64
from datetime import datetime
import pandas as pd
import cv2
from bson.binary import Binary

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DB_NAME')]

EXCEL_FILE = 'attendance.xlsx'

def save_face_encoding(name, face_encoding, frame):
    """Save face encoding and image to both database and file system"""
    # Save to MongoDB
    users_collection = db.users
    
    # Convert face encoding to bytes
    face_encoding_bytes = pickle.dumps(face_encoding)
    
    # Convert image directly to binary
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_binary = Binary(img_encoded.tobytes())
    
    user_data = {
        'name': name,
        'face_encoding': face_encoding_bytes,
        'image_data': img_binary,  # Store as binary data
        'created_at': datetime.now()
    }
    
    # Check if user already exists in MongoDB
    existing_user = users_collection.find_one({'name': name})
    if existing_user:
        users_collection.update_one(
            {'name': name},
            {'$set': user_data}
        )
    else:
        users_collection.insert_one(user_data)
    
    # Save face image to file system as well
    cv2.imwrite(f'known_faces/{name}.jpg', frame)

def get_user_image(name):
    """Retrieve user's image from database"""
    users_collection = db.users
    user = users_collection.find_one({'name': name})
    if user and 'image_data' in user:
        # Convert binary data back to image
        nparr = np.frombuffer(user['image_data'], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    return None

def get_all_face_encodings():
    """Retrieve face encodings from database"""
    encodings = []
    names = []
    
    # Get from MongoDB
    users_collection = db.users
    users = users_collection.find()
    
    for user in users:
        try:
            # Convert bytes back to face encoding and ensure it's the correct shape
            face_encoding = pickle.loads(user['face_encoding'])
            if isinstance(face_encoding, np.ndarray) and face_encoding.size == 128:
                # Ensure the encoding is a 1D array of 128 dimensions
                face_encoding = face_encoding.reshape(128,)
                encodings.append(face_encoding)
                names.append(user['name'])
        except Exception as e:
            print(f"Error loading face encoding for {user['name']}: {e}")
            continue
    
    return np.array(encodings), names

def mark_attendance_db(name):
    """Mark attendance in both MongoDB and Excel"""
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    
    # Mark in MongoDB
    attendance_collection = db.attendance
    existing_attendance = attendance_collection.find_one({
        'name': name,
        'date': today
    })
    
    if not existing_attendance:
        attendance_collection.insert_one({
            'name': name,
            'date': today,
            'time': now,
            'created_at': datetime.now(),
            'Name': name,  # Add standardized fields
            'Date': today,
            'Time': now
        })
        
        # Mark in Excel
        try:
            if os.path.exists(EXCEL_FILE):
                df = pd.read_excel(EXCEL_FILE)
            else:
                df = pd.DataFrame(columns=["Name", "Date", "Time"])
            
            # Check if entry already exists in Excel
            if not ((df["Name"] == name) & (df["Date"] == today)).any():
                new_row = pd.DataFrame([[name, today, now]], columns=["Name", "Date", "Time"])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_excel(EXCEL_FILE, index=False)
        except Exception as e:
            print(f"Error writing to Excel: {e}")

def get_attendance_records():
    """Get attendance records from both MongoDB and Excel"""
    records = []
    
    # Get from MongoDB
    attendance_collection = db.attendance
    mongo_records = list(attendance_collection.find({}, {'_id': 0}).sort('created_at', -1))
    
    # Standardize MongoDB records
    for record in mongo_records:
        records.append({
            'Name': record.get('name', 'Unknown'),
            'Date': record.get('date', ''),
            'Time': record.get('time', '')
        })
    
    # Get from Excel
    try:
        if os.path.exists(EXCEL_FILE):
            excel_df = pd.read_excel(EXCEL_FILE)
            # Ensure column names are correct
            if all(col in excel_df.columns for col in ['Name', 'Date', 'Time']):
                excel_records = excel_df.to_dict('records')
                records.extend(excel_records)
    except Exception as e:
        print(f"Error reading Excel: {e}")
    
    # Deduplicate records
    seen = set()
    unique_records = []
    
    for record in records:
        key = (record['Name'], record['Date'])
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
    
    # Sort by date and time
    unique_records.sort(key=lambda x: (x['Date'], x['Time']), reverse=True)
    return unique_records
