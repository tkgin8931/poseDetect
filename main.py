import tensorflow as tf
import tensorflow_hub as hub 
import numpy as np
import cv2 

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

movement = model.signatures['serving_default']

def detect_pose(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame,axis=0),256,256)
    input_img = tf.cast(img, dtype=tf.int32)
    results = movement(input_img)
    keypoints = results['output_0'].numpy()
    return keypoints

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    keypoints = detect_pose(frame)
    cv2.imshow('MoveNet Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()
