import cv2
import os
import datetime
import time
import webbrowser
import random
import smtplib
from email.mime.text import MIMEText
from flask import Flask, render_template
import onnxruntime as ort
import numpy as np

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- PATHS ----------------
CAPTURE_DIR = "static/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    "models/haarcascade_smile.xml"
)

# Age model
age_net = cv2.dnn.readNetFromCaffe(
    "models/deploy_age.prototxt",
    "models/age_net.caffemodel"
)
AGE_BUCKETS = [
    "0-2", "4-6", "8-12", "15-20",
    "25-32", "38-43", "48-53", "60+"
]

# Mask ONNX model (320x512)
mask_sess = ort.InferenceSession(
    "models/mask_detector.onnx",
    providers=["CPUExecutionProvider"]
)

# ---------------- DATA ----------------
RESULT = {
    "image": None,
    "emotion": None,
    "smile": None,
    "age": None,
    "mask": None,
    "song": None,
    "joke": None
}

JOKES = [
    "Why don’t scientists trust atoms? Because they make up everything 😄",
    "Why did the computer catch a cold? It left its Windows open 🪟",
    "Why was the math book sad? Because it had too many problems 📘"
]

PEACEFUL_SONG = "https://www.youtube.com/embed/2OEL4P1Rz04"

# ---------------- EMAIL ----------------
def send_email_alert(emotion):
    sender_email = "isuhanisaini@gmail.com"
    app_password = "wszf cxyq gxza lqen"
    guardian_email = "isuhanisaini@gmail.com"

    msg = MIMEText(f"Alert! Emotion detected: {emotion}")
    msg["Subject"] = "Emotion Alert"
    msg["From"] = sender
    msg["To"] = guardian

    try:
        s = smtplib.SMTP("smtp.gmail.com", 587)
        s.starttls()
        s.login(sender, app_password)
        s.send_message(msg)
        s.quit()
    except:
        pass

def expand_face(frame, x, y, w, h, scale=1.4):
    cx, cy = x + w//2, y + h//2
    nw, nh = int(w*scale), int(h*scale)
    return frame[
        max(cy-nh//2,0):min(cy+nh//2,frame.shape[0]),
        max(cx-nw//2,0):min(cx+nw//2,frame.shape[1])
    ]

def preprocess_mask(face):
    face = cv2.resize(face, (512,320))
    face = face.astype("float32")/255.0
    face = face[:,:,::-1]
    face = np.transpose(face,(2,0,1))
    return face.reshape(1,3,320,512)

def detect_emotion(face_gray, face_area, smile, h):
    if smile:
        return "Happy"

    mouth_open_ratio = h / max(face_area**0.5, 1)
    if mouth_open_ratio > 1.15:
        return "Fear"

    if face_area > 26000:
        return "Angry"

    if face_area < 11000:
        return "Sad"

    return "Neutral"

def run_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame,(640,480))
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.1,5,minSize=(80,80))
        if len(faces)==0:
            cv2.imshow("Camera",frame)
            cv2.waitKey(1)
            continue

        x,y,w,h = faces[0]
        face_gray = gray[y:y+h,x:x+w]
        face_color = frame[y:y+h,x:x+w]
        face_area = w*h

        smile = len(smile_cascade.detectMultiScale(face_gray,1.6,20))>0
        emotion = detect_emotion(face_gray, face_area, smile, h)

        blob = cv2.dnn.blobFromImage(
            face_color,1.0,(227,227),
            (78.42,87.76,114.89)
        )
        age_net.setInput(blob)
        age = AGE_BUCKETS[age_net.forward()[0].argmax()]

        expanded = expand_face(frame,x,y,w,h)
        mask_out = mask_sess.run(
            None,{mask_sess.get_inputs()[0].name:preprocess_mask(expanded)}
        )
        mask = "Mask" if float(mask_out[0].mean())<0.35 else "No Mask"

        fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg"
        cv2.imwrite(os.path.join(CAPTURE_DIR,fname),frame)

        RESULT.update({
            "image":"captures/"+fname,
            "emotion":emotion,
            "smile":"Smiling" if smile else "Not Smiling",
            "age":age,
            "mask":mask,
            "song":PEACEFUL_SONG if emotion=="Angry" else None,
            "joke":random.choice(JOKES) if emotion=="Sad" else None
        })

        if emotion=="Fear":
            send_email_alert(emotion)

        cv2.imshow("Camera",frame)
        cv2.waitKey(1500)
        break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def result():
    return render_template("result.html",data=RESULT)

if __name__=="__main__":
    run_camera()
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False)
