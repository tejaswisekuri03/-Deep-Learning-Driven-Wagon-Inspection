import streamlit as st
import cv2
import numpy as np
import torch
import telepot
from telepot.loop import MessageLoop
import easyocr
import pandas as pd

# Initialize the Telegram bot
token = '6893108206:AAF9TMncM6QlaXgVj-XIJwzK6Bc3KWGh67A'
bot = telepot.Bot(token)
id = 1787357340

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

# Initialize the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize a dictionary to store colors for different classes
class_colors = {
    "closed_door": (0, 255, 0),  # Green for "closed_door" class
    "opened_door": (0, 0, 255),  # Red for "opened_door" class
    "wagon_number": (255, 0, 0),  # Blue for "wagon_number" class
    "train": (0, 255, 255),  # Yellow for "train" class
}

# Initialize EasyOCR for wagon number recognition
reader = easyocr.Reader(['en'], gpu=False)

# Initialize a list to store detected wagon numbers
wagon_numbers = []

# Initialize a Pandas DataFrame to store the wagon numbers
wagon_numbers_df = pd.DataFrame(columns=['Wagon Number'])

# Main loop for capturing frames and performing object detection
while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Check if the frame is empty
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        continue

    # Resize the frame to a suitable size for the model
    frame = cv2.resize(frame, (1280, 720))

    # Perform object detection on the frame
    results = model(frame)

    # Convert the detections to NumPy array
    detections = np.array(results.xyxy[0].cpu())

    # Filter the detections to only include the 'opened_door' and 'wagon_number' classes
    opened_door_detections = detections[detections[:, 5] == 1.0]
    wagon_number_detections = detections[detections[:, 5] == 2.0]

    # Check if any opened doors or wagon numbers were detected
    if len(opened_door_detections) > 0 or len(wagon_number_detections) > 0:
        # Draw bounding boxes around the detected objects
        for x1, y1, x2, y2, conf, cls in detections:
            # Get the class label as a string
            class_label = model.names[int(cls)]

            # Get the color for the class label
            color = class_colors[class_label]

            # Draw the bounding box with the specified color
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Put the class label and confidence score on the frame
            cv2.putText(frame, f'{class_label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # If it's a wagon number, extract the wagon number using OCR
            if class_label == 'wagon_number':
                cropped_wagon_number = frame[int(y1):int(y2), int(x1):int(x2)]
                wagon_number_text = reader.readtext(cropped_wagon_number)

                # Add the detected wagon number to the list and DataFrame
                if len(wagon_number_text) > 0:
                    wagon_number = wagon_number_text[0][1]
                    wagon_numbers.append(wagon_number)
                    wagon_numbers_df.loc[len(wagon_numbers_df)] = {'Wagon Number': wagon_number}

        # Trigger captured frames to Telegram when opendoor class got detected by capturing frame
        if len(opened_door_detections) > 0:
            cv2.imwrite('detected_frame_with_bounding_boxes.jpg', frame)

            # Send the captured frame to Telegram
            bot.sendPhoto(id, open('detected_frame_with_bounding_boxes.jpg', 'rb'))

    # Draw bounding boxes around all detected objects
    for x1, y1, x2, y2, conf, cls in detections:
        # Get the class label as a string
        class_label = model.names[int(cls)]

        # Get the color for the class label
        color = class_colors[class_label]

        # Draw the bounding box with the specified color
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Put the class label and confidence score on the frame
        cv2.putText(frame, f'{class_label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with detected objects
    cv2.imshow('Live Camera Feed', frame)

    # Display the frame with detected objects
    # st.image(frame, channels="BGR")

    # Display the detected wagon numbers
    st.write("Detected Wagon Numbers:")
    for wagon_number in wagon_numbers:
        st.write(f"- {wagon_number}")

    # Check for user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the DataFrame to an Excel sheet
wagon_numbers_df.to_excel('wagon_numbers.xlsx', index=False)

# Release the video capture
cap.release()

# Close the connection to Telegram
bot.stop()

# Destroy all windows
cv2.destroyAllWindows()