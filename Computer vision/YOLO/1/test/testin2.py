import cv2
from ultralytics import YOLO
import time
import numpy as np

# Define initial values for ROI coordinates
roi_points = []
roi_selected = False  # True if ROI is selected

model = YOLO('yolov5s.pt')

# Detect objects from classes 0 and 1 only
classes = [0]  # Assuming classes 0 and 1 represent people

# Set the confidence threshold
conf_thresh = 0.4

# IP Camera URL
camera_url = ''

# Text font and color for detection alerts
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
alert_color = (0, 0, 255)  # Red
display_width = 800  # Example width
display_height = 600  # Example height

# Mouse callback function
def draw_roi(event, x, y, flags, param):
    global roi_points, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        roi_selected = True

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

cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Select ROI', 800, 600)
cv2.setMouseCallback('Select ROI', draw_roi)

while not roi_selected:
    ret, frame = cap.read()

    # Draw current polygon
    if len(roi_points) > 1:
        cv2.polylines(frame, [np.array(roi_points)], True, (255, 0, 0), 2)

    cv2.imshow('Select ROI', frame)
    cv2.waitKey(1)

cv2.destroyWindow('Select ROI')

# Convert the ROI points to numpy array
roi_polygon = np.array(roi_points)

while True:
    _, img = cap.read()

    results = model.predict(source=img, save=False, classes=classes, conf=conf_thresh)

    # Draw ROI polygon on the frame
    if len(roi_polygon) > 1:
        cv2.polylines(img, [roi_polygon], True, (255, 0, 0), 2)

    if results:  # Check if any detections were made
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format

                # Check if center of bounding box is within ROI
                box_center_x = (b[0] + b[2]) / 2
                box_center_y = (b[1] + b[3]) / 2
                pt = (int(box_center_x), int(box_center_y))
                if cv2.pointPolygonTest(roi_polygon, pt, False) >= 0:
                    c = box.cls

                    # Draw bounding box in red
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)

                    # Display person detected alert
                    text = f"Person detected in ROI!"
                    cv2.putText(img, text, (roi_polygon[:, 0].min() + 5, roi_polygon[:, 1].min() - 5),
                                font, font_scale, alert_color, font_thickness)

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
