from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import atexit
import torch
import os

app = Flask(__name__)

# ---- Load YOLOv8 model (weights-only recommended) ----
# Option 1: Load full checkpoint (trusted file)
# model = YOLO("best.pt")

# Option 2: Safer: load weights-only checkpoint (recommended)
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found. Make sure it's uploaded!")

model = YOLO(model_path)
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
model = YOLO(ckpt)

# ---- Open webcam ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not start webcam.")

# ---- Ensure webcam is released on exit ----
@atexit.register
def cleanup():
    cap.release()

# ---- Frame generator ----
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame, conf=0.25, iou=0.45)
        annotated_frame = results[0].plot()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---- Run app ----
if __name__ == "__main__":
    app.run(debug=True)
