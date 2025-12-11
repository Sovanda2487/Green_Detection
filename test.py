import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Load your trained CNN model ---
model = load_model("green_detector.h5")

# --- Image size for your model (from training) ---
IMG_SIZE = 128  # adjust if you used 96x96

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("✅ Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed.")
        break

    # --- Preprocess frame for CNN ---
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, IMG_SIZE, IMG_SIZE, 3)

    # --- Predict using CNN ---
    pred = model.predict(img_array, verbose=0)[0][0]  # binary output
    confidence = float(pred)

    # --- Threshold for positive detection ---
    if confidence > 0.5:
        # Use HSV to get the green region for bounding box (Option A)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest green contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Green {confidence:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Show the frame ---
    cv2.imshow("Green Detection (CNN)", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
