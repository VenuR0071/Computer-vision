from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import time

model = YOLO('yolov8x.pt')
#cap = cv2.VideoCapture('rtsp://admin:Admin123@192.168.29.99:554/Streaming/Channels/1')
display_width = 800 # Example width
display_height = 600  # Example height

# Detect objects from classes 0 and 1 only
classes = [0, 1]

# Set the confidence threshold
conf_thresh = 0.7
# IP Camera URL
camera_url = 'rtsp://admin:Admin123@192.168.29.99:554/Streaming/Channels/1'



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

    if _:

        for r in results:

            annotator = Annotator(img)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])

        img = annotator.result()
        frame_resized = cv2.resize(img, (display_width, display_height))
        cv2.imshow('YOLO V8 Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    else:
        print("Lost connection to the camera. Attempting to reconnect...")
        # Release the capture object and try to reconnect
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
