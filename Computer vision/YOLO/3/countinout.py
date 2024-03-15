import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
classes_to_count = [0, 2]  # person and car classes for count

# Function to select ROI interactively
def select_roi(image):
    roi_points = []

    # Calculate the aspect ratio of the input image
    aspect_ratio = image.shape[1] / image.shape[0]

    def on_mouse(event, x, y, flags, param):
        nonlocal roi_points
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            if len(roi_points) > 1:
                cv2.line(image, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
            cv2.imshow('Select ROI', image)

    # Calculate the height of the window based on the aspect ratio
    window_height = int(800 / aspect_ratio)

    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select ROI', 800, window_height)
    cv2.setMouseCallback('Select ROI', on_mouse)

    while True:
        frame_copy = image.copy()
        if len(roi_points) > 1:
            cv2.polylines(frame_copy, [np.array(roi_points)], True, (0, 255, 0), 2)
        cv2.imshow('Select ROI', frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            image = image.copy()
            roi_points = []

    cv2.destroyAllWindows()

    return roi_points


# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Capture a frame to select ROI
cap = cv2.VideoCapture('your IP camera url ')
assert cap.isOpened(), "Error reading video file"
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Failed to capture frame from video.")

# Select ROI interactively
roi_points = select_roi(frame)
print("ROI points:", roi_points)

# Initialize Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=False,
                 reg_pts=roi_points,
                 classes_names=model.names,
                 draw_tracks=True)

# Define static width and height
static_width = 1366
static_height = 768

# Process video frames
cap = cv2.VideoCapture('your IP camera url')
frame_skip = 5  # Number of frames to skip
buffer_size = 5  # Number of frames to buffer
frame_buffer = []

while cap.isOpened():
    for _ in range(buffer_size):
        success, frame = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        frame_buffer.append(frame)

    for idx, frame in enumerate(frame_buffer):
        if idx % frame_skip == 0:
            # Draw ROI on the frame
            for i in range(len(roi_points) - 1):
                cv2.line(frame, roi_points[i], roi_points[i+1], (0, 255, 0), 2)
            if len(roi_points) > 1:
                cv2.line(frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)

            # Object tracking
            tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)

            # Count objects within ROI and draw on frame
            frame = counter.start_counting(frame, tracks)

            # Calculate new height based on aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_height = int(static_width / aspect_ratio)

            # Resize frame while maintaining aspect ratio
            frame = cv2.resize(frame, (static_width, new_height))

            # Add black bars to fill remaining height
            border = (static_height - new_height) // 2
            frame = cv2.copyMakeBorder(frame, border, border, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Show processed frame
            cv2.imshow('Processed Frame', frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    frame_buffer = []  # Clear the frame buffer

# Release resources
cap.release()
cv2.destroyAllWindows()
