# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 01:06:21 2024

@author: vedang
"""
import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\vedan\anaconda3\Lib\site-packages\cv2\typing\DETECT_CRACK.keras')

def preprocess_frame(frame):
    # Resize the frame to the input size expected by your model
    input_size = (128, 128)  # Example size, adjust based on your model's input
    resized_frame = cv2.resize(frame, input_size)
    
    # Normalize the image (optional, depending on your model's training)
    normalized_frame = resized_frame / 255.0
    
    # Add batch dimension
    input_frame = np.expand_dims(normalized_frame, axis=0)
    
    return input_frame
def live_detection():
    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 is the default camera, change if you have multiple cameras

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame
        input_frame = preprocess_frame(frame)

        # Make a prediction
        prediction = model.predict(input_frame)

        # Interpret the prediction (assuming binary classification: 0 for uncracked, 1 for cracked)
        label = 'Cracked' if prediction[0] > 0.5 else 'Uncracked'

        # Display the prediction on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Live Crack Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run live detection
live_detection()
