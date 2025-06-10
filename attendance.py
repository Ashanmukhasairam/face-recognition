from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import csv
import os


# Removed sound functionality as per user request
# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Removed background image as per user request
# imgBackground=cv2.imread("background.png")

attendance_file = 'attendance.csv'

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    # Check if file exists, if not create and write header
    if not os.path.exists(attendance_file):
        with open(attendance_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['NAME', 'TIME'])
    # Read existing entries to avoid duplicates for the same day
    with open(attendance_file, mode='r', newline='') as f:
        reader = csv.reader(f)
        existing_entries = list(reader)
    today = now.date()
    # Check if name already marked today
    for row in existing_entries[1:]:
        if row[0] == name and datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S').date() == today:
            return
    # Append new attendance record
    with open(attendance_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, dt_string])

while True:
    ret,frame=video.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w]
        gray_crop=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img=cv2.resize(gray_crop, (50,75))
        flattened_img=resized_img.flatten().reshape(1,-1)
        label=knn.predict(flattened_img)[0]
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        mark_attendance(label)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==27 or k==48:  # ESC key or '0' key to exit
        break
video.release()
cv2.destroyAllWindows()
