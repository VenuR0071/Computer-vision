import cv2
from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

class RegionCounter:
    def __init__(self):
        self.track_history = defaultdict(list)
        self.counting_regions = [
            {
                "name": "YOLOv8 Polygon Region",
                "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
                "counts": 0,
                "dragging": False,
                "region_color": (255, 42, 4),
                "text_color": (255, 255, 255),
            },
            {
                "name": "YOLOv8 Rectangle Region",
                "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
                "counts": 0,
                "dragging": False,
                "region_color": (37, 255, 225),
                "text_color": (0, 0, 0),
            },
        ]

    def mouse_callback(self, event, x, y, flags, param):
        """
        Handles mouse events for region manipulation.
        """
        current_region = None

        # Mouse left button down event
        if event == cv2.EVENT_LBUTTONDOWN:
            for region in self.counting_regions:
                if region["polygon"].contains(Point((x, y))):
                    current_region = region
                    current_region["dragging"] = True
                    current_region["offset_x"] = x
                    current_region["offset_y"] = y

        # Mouse move event
        elif event == cv2.EVENT_MOUSEMOVE:
            if current_region is not None and current_region["dragging"]:
                dx = x - current_region["offset_x"]
                dy = y - current_region["offset_y"]
                current_region["polygon"] = Polygon(
                    [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
                )
                current_region["offset_x"] = x
                current_region["offset_y"] = y

        # Mouse left button up event
        elif event == cv2.EVENT_LBUTTONUP:
            if current_region is not None and current_region["dragging"]:
                current_region["dragging"] = False

    def run(self, source=None, view_img=False, save_img=False):
        vid_frame_count = 0

        # Video setup for RTSP stream
        stream = cv2.VideoCapture(source)

        if not stream.isOpened():
            print("Error: Cannot open video stream.")
            return

        # Iterate over video frames
        while True:
            success, frame = stream.read()

            if not success:
                print("Error: Cannot read frame.")
                break

            # Draw regions (Polygons/Rectangles)
            for region in self.counting_regions:
                region_label = str(region["counts"])
                region_color = region["region_color"]
                region_text_color = region["text_color"]

                polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

                text_size, _ = cv2.getTextSize(
                    region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
                )
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2
                cv2.rectangle(
                    frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    region_color,
                    -1,
                )
                cv2.putText(
                    frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2
                )
                cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=4)

            if view_img:
                cv2.imshow("Region Counter", frame)

            if save_img:
                # Save frames
                pass

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = RegionCounter()
    counter.run(source="C:\\Users\\Venu\\Downloads\\output_video.mp4", view_img=True, save_img=False)
