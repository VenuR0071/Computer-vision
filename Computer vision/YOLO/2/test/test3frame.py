import cv2
import numpy as np
from threading import Thread

from ultralytics import YOLO

# Define initial values for ROI coordinates
roi_points = []
roi_selected = False  # True if ROI is selected

model = YOLO('yolov8s.pt')

# Detect objects from classes 0 and 1 only
classes = [0]  # Assuming classes 0 and 1 represent people

# Set the confidence threshold
conf_thresh = 0.3

# IP Camera URL
camera_url = 'rtsp://admin:Admin123@192.168.29.99:554/Streaming/Channels/1'


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
frameskip = 4  # Controls how many frames to skip between processing

def read_frames(cap, frames_buffer):
  count = 0  # Counter to track frames read
  while True:
      ret, frame = cap.read()
      if ret:
          if count % (frameskip + 1) == 0:  # Process frame only if count is a multiple of (frameskip + 1)
              frames_buffer.append(frame)
          count += 1


def main():
    global roi_selected

    # Attempt to connect to the IP camera
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Failed to connect to the camera.")
        return

    print("Connected to the camera successfully.")

    frames_buffer = []

    # Start thread to read frames
    frame_reader = Thread(target=read_frames, args=(cap, frames_buffer))
    frame_reader.daemon = True
    frame_reader.start()

    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select ROI', 800, 600)
    cv2.setMouseCallback('Select ROI', draw_roi)

    while not roi_selected:
        if frames_buffer:
            frame = frames_buffer.pop(0)
            if len(roi_points) > 1:
                cv2.polylines(frame, [np.array(roi_points)], True, (255, 0, 0), 2)
            cv2.imshow('Select ROI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow('Select ROI')

    # Convert the ROI points to numpy array
    roi_polygon = np.array(roi_points)

    while True:
        if frames_buffer:
            frame = frames_buffer.pop(0)

            # Get original frame dimensions
            original_height, original_width = frame.shape[:2]

            # Calculate scaling factor to maintain aspect ratio
            scale_factor = min(display_width / original_width, display_height / original_height)

            # Resize frame with the calculated scaling factor
            resized_width = int(original_width * scale_factor)
            resized_height = int(original_height * scale_factor)
            img_resized = cv2.resize(frame, (resized_width, resized_height))

            results = model.predict(source=img_resized, save=False, classes=classes, conf=conf_thresh)

            # Draw ROI polygon on the resized frame
            if len(roi_polygon) > 1:
                scaled_roi_polygon = (roi_polygon * scale_factor).astype(int)
                cv2.polylines(img_resized, [scaled_roi_polygon], True, (255, 0, 0), 2)

            if results:  # Check if any detections were made
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format

                        # Check if center of bounding box is within ROI
                        box_center_x = (b[0] + b[2]) / 2
                        box_center_y = (b[1] + b[3]) / 2
                        pt = (int(box_center_x), int(box_center_y))
                        if cv2.pointPolygonTest(scaled_roi_polygon, pt, False) >= 0:
                            c = box.cls

                            # Draw bounding box in red
                            cv2.rectangle(img_resized, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)

                            # Display person detected alert
                            text = f"Person detected in ROI!"
                            cv2.putText(img_resized, text, (scaled_roi_polygon[:, 0].min() + 5, scaled_roi_polygon[:, 1].min() - 5),
                                        font, font_scale, alert_color, font_thickness)

            cv2.imshow('YOLO V8 Detection', img_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
