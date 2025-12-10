from tensorflow.keras.models import load_model

model = load_model("green_detector.h5")
model.summary()
