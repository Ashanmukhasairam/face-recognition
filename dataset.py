import cv2
import pickle
import numpy as np
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_data=[]

i=0

import os

name=input("Enter Your Name (enter 0 to stop): ")

# Delete existing data files to reset data
if os.path.exists('data/names.pkl'):
    os.remove('data/names.pkl')
if os.path.exists('data/faces_data.pkl'):
    os.remove('data/faces_data.pkl')

while True:
    if name == "0":
        print("Stopping photo capture as requested.")
        break
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w]
        gray_crop=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img=cv2.resize(gray_crop, (50,75))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if len(faces_data)==50:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(len(faces_data), -1)



if 'names.pkl' not in os.listdir('data/'):
    names=[name]*len(faces_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    # Ensure names and faces have the same length
    min_len = min(len(names), len(faces))
    names = names[:min_len]
    faces = faces[:min_len]
    # Save truncated arrays back to files
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
    # Append new data
    names = names + [name]*len(faces_data)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)


