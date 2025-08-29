class CameraPoseEstimator:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_and_match_keypoints(self, frame1, frame2):
        kp1 = self.orb.detect(frame1, None)
        kp1, des1 = self.orb.compute(frame1, kp1)
        kp2 = self.orb.detect(frame2, None)
        kp2, des2 = self.orb.compute(frame2, kp2)
        if des1 is None or des2 is None:
            return [], [], []
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return pts1, pts2, matches

    def estimate_pose(self, frame1, frame2, K=None):
        pts1, pts2, matches = self.extract_and_match_keypoints(frame1, frame2)
        if len(pts1) < 8:
            return None, None, None
        E, mask = cv2.findEssentialMat(pts1, pts2, K if K is not None else 1.0, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None, None
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K if K is not None else 1.0)
        return R, t, mask_pose

class RecognitionAlgorithmManager:
    def __init__(self, algorithm="movenet"):
        self.algorithm = algorithm
        # MoveNetモデルは初期化時にロード
        if algorithm == "movenet":
            self.model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
            self.movement = self.model.signatures['serving_default']
        else:
            self.model = None
            self.movement = None

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        if algorithm == "movenet":
            self.model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
            self.movement = self.model.signatures['serving_default']
        else:
            self.model = None
            self.movement = None

    def recognize(self, frame):
        if self.algorithm == "movenet":
            img = tf.image.resize_with_pad(tf.expand_dims(frame,axis=0),256,256)
            input_img = tf.cast(img, dtype=tf.int32)
            results = self.movement(input_img)
            keypoints = results['output_0'].numpy()
            return {"type": "pose", "keypoints": keypoints}
        elif self.algorithm == "orb":
            orb = cv2.ORB_create()
            kp = orb.detect(frame, None)
            kp, des = orb.compute(frame, kp)
            return {"type": "keypoints", "keypoints": kp}
        elif self.algorithm == "kcf":
            # KCF追跡は初期化とROIが必要。ここではダミー実装
            tracker = cv2.TrackerKCF_create()
            # ROIは外部から指定する必要あり
            # bbox = (x, y, w, h)
            # ok = tracker.init(frame, bbox)
            return {"type": "tracker", "tracker": tracker}
        else:
            return None

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
