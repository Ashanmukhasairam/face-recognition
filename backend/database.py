import logging
from pymongo import MongoClient, errors
from dotenv import load_dotenv
import os
import numpy as np
import pickle
import base64
from datetime import datetime
import pandas as pd
import cv2
from bson.binary import Binary
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
EXCEL_FILE = os.path.join(os.path.dirname(__file__), 'attendance.xlsx')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), 'known_faces')

# Ensure directories exist
for directory in [DATA_DIR, KNOWN_FACES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize Excel file if it doesn't exist
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(EXCEL_FILE, index=False)

# Global variables
client = None
db = None
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def init_db():
    """Initialize database connection with retry logic"""
    global client, db
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable is not set")
    
    db_name = os.getenv('DB_NAME', 'face_recognition')
    
    for attempt in range(MAX_RETRIES):
        try:
            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            client.server_info()
            db = client[db_name]
            logger.info(f"Successfully connected to MongoDB database: {db_name}")
            return
        except errors.ServerSelectionTimeoutError as e:
            if attempt == MAX_RETRIES - 1:
                raise Exception(f"Could not connect to MongoDB after {MAX_RETRIES} attempts: {e}")
            logger.warning(f"MongoDB connection attempt {attempt + 1} failed, retrying...")
            time.sleep(RETRY_DELAY)

def db_operation(f):
    """Decorator to handle database operations with retry logic"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return f(*args, **kwargs)
            except errors.AutoReconnect as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to execute database operation after {MAX_RETRIES} attempts")
                    raise
                logger.warning(f"Database operation failed, attempt {attempt + 1}, retrying...")
                time.sleep(RETRY_DELAY)
    return wrapper

@db_operation
def save_face_encoding(name, face_encoding, frame):
    """Save face encoding and image to database with error handling"""
    try:
        users_collection = db.users
        
        # Convert face encoding to bytes
        face_encoding_bytes = pickle.dumps(face_encoding)
        
        # Convert image directly to binary
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_binary = Binary(img_encoded.tobytes())
        
        user_data = {
            'name': name,
            'face_encoding': face_encoding_bytes,
            'image_data': img_binary,
            'created_at': datetime.now()
        }
        
        # Check if user already exists
        existing_user = users_collection.find_one({'name': name})
        if existing_user:
            users_collection.update_one(
                {'name': name},
                {'$set': user_data}
            )
            logger.info(f"Updated face encoding for user: {name}")
        else:
            users_collection.insert_one(user_data)
            logger.info(f"Saved new face encoding for user: {name}")
            
        # Save face image to file system as backup
        try:
            os.makedirs('known_faces', exist_ok=True)
            cv2.imwrite(f'known_faces/{name}.jpg', frame)
        except Exception as e:
            logger.warning(f"Could not save backup image to file system: {e}")
            
    except Exception as e:
        logger.error(f"Error saving face encoding: {e}")
        raise

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
            logger.error(f"Error loading face encoding for {user['name']}: {e}")
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
            logger.error(f"Error writing to Excel: {e}")

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
        logger.error(f"Error reading Excel: {e}")
    
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

# Initialize the database connection
init_db()
