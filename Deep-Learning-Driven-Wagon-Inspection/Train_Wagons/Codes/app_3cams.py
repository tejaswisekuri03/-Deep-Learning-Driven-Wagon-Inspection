import streamlit as st
import cv2
import numpy as np
import torch
import telepot
from telepot.loop import MessageLoop


# Initialize the Telegram bot
token = '6808877124:AAHHTGTTfLfmKNaYxfoosyIwDq1igMQje6s'
bot = telepot.Bot(token)
id = 1420686425

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Initialize the video captures for 3 webcams
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap3 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Initialize variables for storing detected objects and synchronization
objects_detected = False
detected_frames = []

# Main loop for capturing frames and performing object detection
while True:
    # Read the frames from the video captures
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    # Resize the frames to a suitable size for the model
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))
    frame3 = cv2.resize(frame3, (640, 480))

    # Perform object detection on the frames
    results1 = model(frame1)
    results2 = model(frame2)
    results3 = model(frame3)

    # Get the detected objects and their bounding boxes for each webcam
    detections1 = results1.xyxy[0]
    detections2 = results2.xyxy[0]
    detections3 = results3.xyxy[0]

    # Filter the detections to only include the 'opendoor' class for each webcam
    opendoor_detections1 = detections1[detections1[:, 5] == 1.0]
    opendoor_detections2 = detections2[detections2[:, 5] == 1.0]
    opendoor_detections3 = detections3[detections3[:, 5] == 1.0]

    # Check if any opendoors were detected in any of the webcams
    if len(opendoor_detections1) > 0 or len(opendoor_detections2) > 0 or len(opendoor_detections3) > 0:
        # Set the flag to indicate that objects were detected
        objects_detected = True

        # Add the detected frames to the list
        detected_frames.append(frame1)
        detected_frames.append(frame2)
        detected_frames.append(frame3)

    # Display the frames with detected objects for each webcam
    for x1, y1, x2, y2, conf, cls in detections1:
        cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame1, f'{cls} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for x1, y1, x2, y2, conf, cls in detections2:
        cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame2, f'{cls} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for x1, y1, x2, y2, conf, cls in detections3:
        cv2.rectangle(frame3, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame3, f'{cls} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frames
    cv2.imshow('Webcam 1', frame1)
    cv2.imshow('Webcam 2', frame2)
    cv2.imshow('Webcam 3', frame3)

    # Check for user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if objects were detected and all three webcams have captured a frame
    if objects_detected and len(detected_frames) == 3:
        # Send the captured frames to Telegram
        for frame in detected_frames:
            bot.sendPhoto(id, frame)

    # Reset the variables for the next iteration
    objects_detected = False
    detected_frames = []

# Release the video captures
cap1.release()
cap2.release()
cap3.release()

# Close the connection to Telegram
bot.stop()

# Destroy all windows
cv2.destroyAllWindows()
