import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, render_template, Response

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

BUFFER_SIZE = 20
feature_buffer = deque(maxlen=BUFFER_SIZE)
fall_counter = 0
FALL_FRAMES = 10

def extract_features(landmarks):
    try:
        LS = landmarks[11]
        RS = landmarks[12]
        LH = landmarks[23]
        RH = landmarks[24]

        sx = (LS.x + RS.x) / 2
        sy = (LS.y + RS.y) / 2
        hx = (LH.x + RH.x) / 2
        hy = (LH.y + RH.y) / 2

        ydiff = abs(sy - hy)
        aspect = abs(sx - hx) / (ydiff if ydiff > 0 else 1)
        angle = np.arctan2(hy - sy, hx - sx)
        height = (sy + hy) / 2

        return aspect, angle, height
    except:
        return 0, 0, 0

def detect_fall(buffer):
    if len(buffer) < BUFFER_SIZE:
        return False

    aspects = [f[0] for f in buffer]
    angles = [f[1] for f in buffer]
    heights = [f[2] for f in buffer]

    sudden_change = np.max(angles) - np.min(angles) > 1.0
    horizontal = sum(a < 0.5 for a in aspects) > BUFFER_SIZE * 0.9
    drop = np.max(heights) - np.min(heights) > 0.20

    return sudden_change and horizontal and drop

def gen_frames():
    global fall_counter
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            feature_buffer.append(extract_features(results.pose_landmarks.landmark))

            if detect_fall(feature_buffer):
                fall_counter += 1
            else:
                fall_counter = max(0, fall_counter - 1)

            if fall_counter >= FALL_FRAMES:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                cv2.putText(frame, "FALL DETECTED!", (50, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
