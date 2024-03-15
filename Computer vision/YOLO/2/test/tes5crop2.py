import cv2
import numpy as np
from threading import Thread
import time
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

    # Draw the ROI polygon dynamically while selecting points
    if len(roi_points) > 1:
        roi_frame = param.copy()
        cv2.polylines(roi_frame, [np.array(roi_points)], False, (255, 0, 0), 2)
        cv2.imshow('Select ROI', roi_frame)

frameskip = 3  # Controls how many frames to skip between processing

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

    start_time = time.time()
    frames_processed = 0

    while not roi_selected:
        if frames_buffer:
            frame = frames_buffer.pop(0)

            # Set mouse callback function to draw ROI dynamically
            cv2.setMouseCallback('Select ROI', draw_roi, param=frame)

            cv2.imshow('Select ROI', frame)

            frames_processed += 1
            elapsed_time = time.time() - start_time
            fps = frames_processed / elapsed_time
            print(f"FPS: {fps:.2f}")

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

            # Calculate scaling factor for width and height separately to maintain aspect ratio
            width_scale_factor = display_width / original_width
            height_scale_factor = display_height / original_height
            scale_factor = min(width_scale_factor, height_scale_factor)

            # Resize frame with the calculated scaling factor
            resized_width = int(original_width * scale_factor)
            resized_height = int(original_height * scale_factor)
            img_resized = cv2.resize(frame, (resized_width, resized_height))

            # Convert the ROI points to numpy array and scale them
            roi_polygon = np.array(roi_points)
            scaled_roi_polygon = (roi_polygon * scale_factor).astype(int)

            # Create a mask to overlay the area outside the ROI
            mask = np.zeros_like(img_resized)
            cv2.fillPoly(mask, [scaled_roi_polygon], (255, 255, 255))
            mask_inv = cv2.bitwise_not(mask)

            # Overlay the area outside the ROI with a semi-transparent gray
            outside_roi = cv2.bitwise_and(img_resized, mask_inv)
            overlay_color = (128, 128, 128)  # Gray
            outside_roi = cv2.addWeighted(outside_roi, 0.5, np.full_like(outside_roi, overlay_color), 0.5, 0)

            # Draw the ROI polygon on the resized frame
            cv2.polylines(outside_roi, [scaled_roi_polygon], True, (255, 0, 0), 2)

            # Crop frame to ROI region
            inside_roi = cv2.bitwise_and(img_resized, mask)

            # Combine the inside and outside ROI frames
            img_with_overlay = cv2.add(outside_roi, inside_roi)

            # Perform object detection only within the ROI
            results = model.predict(source=inside_roi, save=False, classes=classes, conf=conf_thresh)

            if results:  # Check if any detections were made
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format

                        # Draw bounding box in red
                        cv2.rectangle(img_with_overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)

                        # Display person detected alert
                        text = f"Person detected in ROI!"
                        cv2.putText(img_with_overlay, text, (10, 30), font, font_scale, alert_color, font_thickness)

            cv2.imshow('YOLO V8 Detection', img_with_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Calculate FPS and display
            frames_processed += 1
            elapsed_time = time.time() - start_time
            fps = frames_processed / elapsed_time
            print(f"FPS: {fps:.2f}")

    # Release capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
