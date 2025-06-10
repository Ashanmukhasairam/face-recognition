import pickle
import os
import cv2
import numpy as np
from database import save_face_encoding
from bson.binary import Binary

def migrate_old_data():
    """Migrate data from pkl files to MongoDB"""
    print("Starting data migration...")
    
    # Check if old data files exist
    if not os.path.exists('../data/faces_data.pkl') or not os.path.exists('../data/names.pkl'):
        print("No old data files found.")
        return
        
    try:
        # Load old data
        with open('../data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        with open('../data/names.pkl', 'rb') as f:
            names = pickle.load(f)
            
        print(f"Found {len(names)} records to migrate.")
        
        # Migrate each face
        for i, (face_encoding, name) in enumerate(zip(faces_data, names)):
            # Check if image exists in old system
            old_image_path = f'../known_faces/{name}.jpg'
            if os.path.exists(old_image_path):
                frame = cv2.imread(old_image_path)
            else:
                # Create a blank image if no image exists
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
            # Save to new system
            save_face_encoding(name, face_encoding, frame)
            print(f"Migrated data for {name}")
            
        print("Migration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == "__main__":
    migrate_old_data()
