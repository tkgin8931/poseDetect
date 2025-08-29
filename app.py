from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movement = model.signatures['serving_default']

def detect_pose(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame,axis=0),256,256)
    input_img = tf.cast(img, dtype=tf.int32)
    results = movement(input_img)
    keypoints = results['output_0'].numpy()
    return keypoints

def draw_keypoints(frame, keypoints):
    import sys
    h, w, _ = frame.shape
    total_kp = 0
    print(f"keypoints shape: {keypoints.shape if hasattr(keypoints, 'shape') else type(keypoints)}")
    # MoveNet MultiPose: keypoints.shape = (1, num_person, 56)
    for person_idx, person in enumerate(keypoints[0]):
        # 17 keypoints: each 3 values (y, x, score)
        scores = []
        for kp_idx in range(17):
            y = person[kp_idx * 3]
            x = person[kp_idx * 3 + 1]
            score = person[kp_idx * 3 + 2]
            scores.append(score)
            if score > 0.2:
                cx = int(x * w)
                cy = int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
                total_kp += 1
        print(f"person {person_idx} scores: {scores}")
        sys.stdout.flush()
    print(f"Drawn keypoints: {total_kp}")
    sys.stdout.flush()
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = detect_pose(frame)
        print(f"keypoints output: {keypoints}")  # デバッグ出力
        frame = draw_keypoints(frame, keypoints)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return redirect(url_for('video', filename=filename))

@app.route('/video/<filename>')
def video(filename):
    return Response(gen_video(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_video(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = detect_pose(frame)
        print(f"keypoints output: {keypoints}") 
        frame = draw_keypoints(frame, keypoints)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
