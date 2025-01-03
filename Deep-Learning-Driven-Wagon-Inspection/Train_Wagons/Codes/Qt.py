import cv2
import numpy as np
import torch
import sys
import qimage2ndarray
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
import easyocr
import pandas as pd
import telepot
from telepot.loop import MessageLoop

# Initialize the Telegram bot
token = '6808877124:AAHHTGTTfLfmKNaYxfoosyIwDq1igMQje6s'
bot = telepot.Bot(token)
id = 1420686425

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

# Initialize the video captures
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Initialize a dictionary to store colors for different classes
class_colors = {
    "closed_door": (0, 255, 0),    # Green for "closed_door" class
    "opened_door": (0, 0, 255),    # Red for "opened_door" class
    "wagon_number": (255, 0, 0),   # Blue for "wagon_number" class
    "train": (0, 255, 255),        # Yellow for "train" class
}

# Initialize EasyOCR for wagon number recognition
reader = easyocr.Reader(['en'], gpu=False)

# Initialize a list to store detected wagon numbers
wagon_numbers = []

# Initialize a Pandas DataFrame to store the wagon numbers
wagon_numbers_df = pd.DataFrame(columns=['Wagon Number'])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Wagon Anomaly Detection")
        self.resize(1600, 600)  # Adjusted window size to accommodate two camera feeds

        # Create a QVBoxLayout as the main layout
        main_layout = QVBoxLayout()

        # Create a QHBoxLayout for the camera feeds and the detected wagon numbers
        hbox = QHBoxLayout()

        # Create labels for camera feeds
        self.cam_label1 = QLabel()
        self.cam_label1.setFixedSize(720, 480)  # Adjusted size for the first camera feed

        self.cam_label2 = QLabel()
        self.cam_label2.setFixedSize(720, 480)  # Adjusted size for the second camera feed

        # Add the camera feeds to the QHBoxLayout
        hbox.addWidget(self.cam_label1)
        hbox.addWidget(self.cam_label2)

        # Add the QHBoxLayout to the main layout
        main_layout.addLayout(hbox)

        # Create a QLabel for the detected wagon numbers
        self.wagon_numbers_label = QLabel()
        self.wagon_numbers_label.setFixedSize(640, 480)

        # Add the QLabel to the main layout
        main_layout.addWidget(self.wagon_numbers_label)

        # Create a QHBoxLayout for the buttons
        button_layout = QHBoxLayout()

        # Create buttons to control camera feeds
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_camera)

        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_camera)

        # Add buttons to the QHBoxLayout
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.continue_button)

        # Add the button layout to the main layout
        main_layout.addLayout(button_layout)

        # Set the main layout as the central widget of the window
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(main_layout)
        self.setCentralWidget(self.centralWidget)

        # Initialize the camera feed timers
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.update_camera_feed1)
        self.timer1.start(100)

        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.update_camera_feed2)
        self.timer2.start(100)

        # Flag to control camera feeds
        self.camera_running = True

    def stop_camera(self):
        # Stop the camera feed timers
        self.timer1.stop()
        self.timer2.stop()
        self.camera_running = False

    def continue_camera(self):
        # Continue the camera feed timers
        self.timer1.start(100)
        self.timer2.start(100)
        self.camera_running = True

    def update_camera_feed1(self):
        if self.camera_running:
            self.update_camera_feed(self.cam_label1, cap1)

    def update_camera_feed2(self):
        if self.camera_running:
            self.update_camera_feed(self.cam_label2, cap2)

    def update_camera_feed(self, label, capture):
        # Read the frame from the video capture
        ret, frame = capture.read()

        # Check if the frame is empty
        if frame is None:
            return

        # Resize the frame to a suitable size for the model
        frame = cv2.resize(frame, (720, 480))

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
                color = class_colors.get(class_label, (0, 0, 0))  # Default color to black if class not found

                # Draw the                # bounding box with the specified color
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

        # Display the frame with detected objects
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(qimage2ndarray.array2qimage(frame))
        label.setPixmap(QPixmap.fromImage(image))

        # Display the detected wagon numbers
        self.wagon_numbers_label.setText('\n'.join(wagon_numbers))

        # Save the wagon numbers to an Excel sheet
        wagon_numbers_df.to_excel('wagon_numbers.xlsx', index=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
