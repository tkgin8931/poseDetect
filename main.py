
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VideoSource:
    def __init__(self, source=0):
        """
        source: 0（デフォルト）でインカメラ、strで動画ファイルパス
        """
        self.cap = cv2.VideoCapture(source)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()      

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movement = model.signatures['serving_default']

def detect_pose(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame,axis=0),256,256)
    input_img = tf.cast(img, dtype=tf.int32)
    results = movement(input_img)
    keypoints = results['output_0'].numpy()
    return keypoints


if len(sys.argv) > 1:
    source = sys.argv[1]
else:
    source = 0
video = VideoSource(source)

while video.is_opened():
    ret, frame = video.read()
    if not ret:
        break
    keypoints = detect_pose(frame)
    h, w, _ = frame.shape
    for person in keypoints[0]:
        for kp in person:
            y, x, score = kp
            if score > 0.3:
                cx = int(x * w)
                cy = int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
    cv2.imshow('MoveNet Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
