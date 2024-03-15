import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import time

# Define ROI (Region of Interest) coordinates
roi_x1, roi_y1, roi_x2, roi_y2 = 120, 90, 680, 510  # Adjust these values as needed

model = YOLO('yolov8x.pt')
display_width = 800  # Example width
display_height = 600  # Example height

# Detect objects from classes 0 and 1 only
classes = [0, 1]  # Assuming classes 0 and 1 represent people

# Set the confidence threshold
conf_thresh = 0.7

# IP Camera URL
camera_url = 'rtsp://admin:Admin123@192.168.29.99:554/Streaming/Channels/1'

# Text font and color for detection alerts
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
alert_color = (0, 0, 255)  # Red

# Retry logic
while True:
    # Attempt to connect to the IP camera
    cap = cv2.VideoCapture(camera_url)

    if cap.isOpened():
        print("Connected to the camera successfully.")
        break
    else:
        print("Failed to connect to the camera. Retrying in 5 seconds...")
        time.sleep(5)

while True:
    _, img = cap.read()

    results = model.predict(source=img, save=False, classes=classes, conf=conf_thresh)

    # Draw ROI rectangle on the frame
    cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    if results:  # Check if any detections were made
        for r in results:

            annotator = Annotator(img)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format

                # Check if bounding box is within ROI
                if roi_x1 < b[0] < roi_x2 and roi_y1 < b[1] < roi_y2:
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)], color=(0, 0, 255))  # Red for ROI detection

                    # Display person detected alert
                    text = f"Person detected in ROI!"
                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_x, text_y = (roi_x1 + 5, roi_y1 - 5)  # Place alert text above ROI
                    cv2.rectangle(img, (text_x, text_y - text_size[1]), (text_x + text_size[0] + 2, text_y), alert_color, cv2.FILLED)
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, alert_color, font_thickness)

        img = annotator.result()

    frame_resized = cv2.resize(img, (display_width, display_height))
    cv2.imshow('YOLO V8 Detection', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

    # Reconnect logic on frame drop
    if not _:
        print("Lost connection to the camera. Attempting to reconnect...")
        cap.release()
        while True:
            cap = cv2.VideoCapture(camera_url)
            if cap.isOpened():
                print("Reconnected to the camera successfully.")
                break
            else:
                print("Failed to reconnect to the camera. Retrying in 5 seconds...")
                time.sleep(5)

# Release capture and close all windows
cap.release()
cv2.destroyAllWindows()
