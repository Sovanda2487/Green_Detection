# file: green_follower_pi4.py

import cv2
import numpy as np
import time
import threading
from tflite_runtime.interpreter import Interpreter
from auppbot import AUPPBot

# -------------------------------
# --- Robot motor setup ----------
# -------------------------------
bot = AUPPBot("/dev/ttyUSB0", 115200, auto_safe=True)

# --- Motor movement functions ---
def move_forward(duration=4):
    bot.motor1.speed(25)
    bot.motor2.speed(25)
    bot.motor3.speed(25)
    bot.motor4.speed(25)
    time.sleep(duration)
    bot.stop_all()

def move_left(duration=4):
    bot.motor1.speed(-25)
    bot.motor2.speed(-25)
    bot.motor3.speed(25)
    bot.motor4.speed(25)
    time.sleep(duration)
    bot.stop_all()

def move_right(duration=4):
    bot.motor1.speed(25)
    bot.motor2.speed(25)
    bot.motor3.speed(-25)
    bot.motor4.speed(-25)
    time.sleep(duration)
    bot.stop_all()

def stop_scan():
    bot.stop_all()
    time.sleep(0.5)

# -------------------------------
# --- TFLite model setup ----------
# -------------------------------
MODEL_PATH = "green_detector.tflite"
interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]['shape'][1]  # assuming square input

# -------------------------------
# --- Camera and detection -------
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("✅ Camera opened. Press Ctrl+C to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_h, frame_w = frame.shape[:2]

        # --- Define inner frame (wider) ---
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

        # --- Run TFLite inference ---
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        confidence = float(pred)

        cx = None
        if confidence > 0.5:
            # HSV mask for green
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

        # --- Determine region ---
        region = "NONE_GREEN"
        inner_w_adj = inner_x2 - inner_x1
        third = inner_w_adj // 3
        if cx is not None:
            if cx < third:
                region = "LEFT"
            elif cx > 2*third:
                region = "RIGHT"
            else:
                region = "MIDDLE"

        # --- Execute robot behavior ---
        if region == "LEFT":
            move_left()
        elif region == "RIGHT":
            move_right()
        elif region == "MIDDLE":
            move_forward()
        else:
            stop_scan()

        # --- Draw visualization ---
        cv2.rectangle(frame, (inner_x1, inner_y1), (inner_x2, inner_y2), (255, 255, 0), 2)
        # Draw inner divisions
        cv2.line(inner_frame, (third,0), (third, inner_h), (255,255,255),2)
        cv2.line(inner_frame, (2*third,0), (2*third, inner_h), (255,255,255),2)
        # Draw centroid
        if cx is not None:
            cv2.circle(inner_frame, (cx, inner_h//2), 5, (0,0,255), -1)
        frame[inner_y1:inner_y2, inner_x1:inner_x2] = inner_frame
        # Display detected region
        cv2.putText(frame, f"Detected: {region}", (inner_x1, inner_y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Green Follower Robot", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

cap.release()
cv2.destroyAllWindows()
bot.stop_all()
