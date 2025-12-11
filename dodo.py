# file: green_follower_laptop_innerframe_labels_outside.py

import cv2
import numpy as np
import threading
from flask import Flask, Response
from tensorflow.keras.models import load_model

# --- Flask app ---
app = Flask(__name__)

# --- Load trained Keras model ---
model = load_model("green_detector.h5")
IMG_SIZE = 128

# --- Global frame for Flask ---
global_frame = None

# --- Camera and detection thread ---
def camera_loop():
    global global_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_h, frame_w = frame.shape[:2]

        # --- Inner frame (wider version) ---
        inner_h = frame_h // 2
        inner_w = frame_w // 2 + 100
        inner_x1 = max(frame_w // 4 - 50, 0)
        inner_y1 = frame_h // 4 + 30
        inner_x2 = min(inner_x1 + inner_w, frame_w)
        inner_y2 = inner_y1 + inner_h

        inner_frame = frame[inner_y1:inner_y2, inner_x1:inner_x2].copy()

        # --- Preprocess for model ---
        img = cv2.resize(inner_frame, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img_rgb.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- Predict green presence ---
        pred = model.predict(img_array, verbose=0)[0][0]
        confidence = float(pred)

        cx = None
        if confidence > 0.5:
            hsv = cv2.cvtColor(inner_frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w // 2
                cv2.rectangle(inner_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- Draw inner frame divisions ---
        inner_w_adj = inner_x2 - inner_x1
        inner_h_adj = inner_y2 - inner_y1
        third = inner_w_adj // 3
        cv2.line(inner_frame, (third, 0), (third, inner_h_adj), (255, 255, 255), 2)
        cv2.line(inner_frame, (2*third, 0), (2*third, inner_h_adj), (255, 255, 255), 2)

        # --- Overlay labels centered inside each part, but outside inner frame ---
        label_y = -10  # negative means above inner frame
        cv2.putText(frame, "LEFT", (inner_x1 + third//2 - 20, inner_y1 + label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "MIDDLE", (inner_x1 + third + third//2 - 35, inner_y1 + label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "RIGHT", (inner_x1 + 2*third + third//2 - 25, inner_y1 + label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Visualize centroid ---
        region = "NONE"
        if cx is not None:
            cv2.circle(inner_frame, (cx, inner_h_adj//2), 5, (0, 0, 255), -1)
            if cx < third:
                region = "LEFT"
            elif cx > 2*third:
                region = "RIGHT"
            else:
                region = "MIDDLE"

        # --- Display detected region above inner frame ---
        cv2.putText(frame, f"Detected: {region}", (inner_x1, inner_y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- Draw outer frame rectangle and paste inner frame ---
        cv2.rectangle(frame, (inner_x1, inner_y1), (inner_x2, inner_y2), (255, 255, 0), 2)
        frame[inner_y1:inner_y2, inner_x1:inner_x2] = inner_frame

        global_frame = frame

# --- Flask streaming ---
def generate_frames():
    global global_frame
    while True:
        if global_frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', global_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Green Detection Laptop Test (Labels Outside Inner Frame)</h1><img src='/video_feed' width='640'/>"

# --- Run camera loop in a thread ---
t = threading.Thread(target=camera_loop, daemon=True)
t.start()

# --- Start Flask server ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
