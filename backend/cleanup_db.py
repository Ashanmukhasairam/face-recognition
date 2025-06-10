from database import client, db
import numpy as np
import pickle

def cleanup_database():
    """Clean up any corrupted face encodings in the database"""
    users_collection = db.users
    users = users_collection.find()
    
    cleaned = 0
    errors = 0
    
    for user in users:
        try:
            # Try to load and validate the face encoding
            face_encoding = pickle.loads(user['face_encoding'])
            if not isinstance(face_encoding, np.ndarray) or face_encoding.size != 128:
                print(f"Invalid encoding for user {user['name']}, removing...")
                users_collection.delete_one({'_id': user['_id']})
                errors += 1
            else:
                # Ensure correct shape and save back
                face_encoding = face_encoding.reshape(128,)
                users_collection.update_one(
                    {'_id': user['_id']},
                    {'$set': {'face_encoding': pickle.dumps(face_encoding)}}
                )
                cleaned += 1
        except Exception as e:
            print(f"Error processing user {user.get('name', 'unknown')}: {e}")
            errors += 1
    
    print(f"Cleanup complete. Processed {cleaned} valid entries, removed {errors} invalid entries.")

if __name__ == "__main__":
    cleanup_database()
