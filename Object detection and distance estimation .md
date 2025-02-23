# Stereo object detector and depth estimation

In this example i have used yolov8 model for the object detection

```python

import cv2
import numpy as np
from ultralytics import YOLO
import time


class StereoObjectDetector:
    def __init__(self, calib_file, left_index=0, right_index=1):
        # Initialize cameras
        self.left_cam = cv2.VideoCapture(left_index)
        self.right_cam = cv2.VideoCapture(right_index)

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')

        # Load calibration parameters from npz file
        self.load_calibration(calib_file)

        # Initialize stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,  # Increased for better range
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=25,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Get stereo rectification maps
        self.init_rectification_maps()

    def load_calibration(self, calib_file):
        # Load the npz file
        calib_data = np.load(calib_file)

        # Extract all parameters
        self.mtx_left = calib_data['mtx_left']
        self.mtx_right = calib_data['mtx_right']
        self.dist_left = calib_data['dist_left']
        self.dist_right = calib_data['dist_right']
        self.R1 = calib_data['R1']
        self.R2 = calib_data['R2']
        self.P1 = calib_data['P1']
        self.P2 = calib_data['P2']
        self.Q = calib_data['Q']

        # Print loaded parameters for verification
        print("Loaded calibration parameters:")
        for key in calib_data.files:
            print(f"{key} shape: {calib_data[key].shape}")

    def init_rectification_maps(self):
        # Get image size from camera
        _, left_frame = self.left_cam.read()
        if left_frame is None:
            raise ValueError("Could not read from left camera")
        img_size = left_frame.shape[:2][::-1]  # width, height

        # Calculate rectification maps
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.mtx_left, self.dist_left, self.R1, self.P1, img_size, cv2.CV_32FC1)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.mtx_right, self.dist_right, self.R2, self.P2, img_size, cv2.CV_32FC1)

    def compute_depth(self, left_rect, right_rect):
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Filter out invalid disparity values
        valid_mask = (disparity >= 0) & (disparity < disparity.max())

        # Create 3D points
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        return points_3d, disparity, valid_mask

    def detect_objects_with_depth(self, left_frame, right_frame):
        # Rectify images
        left_rect = cv2.remap(left_frame, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        # Compute depth
        points_3d, disparity, valid_mask = self.compute_depth(left_rect, right_rect)

        # Run YOLO detection on left image
        results = self.model(left_rect, stream=True)

        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence and class
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = self.model.names[class_id]

                # Calculate center point of the box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Get average depth in a small window around center point
                window_size = 5
                y_start = max(0, center_y - window_size)
                y_end = min(points_3d.shape[0], center_y + window_size)
                x_start = max(0, center_x - window_size)
                x_end = min(points_3d.shape[1], center_x + window_size)

                depth_window = points_3d[y_start:y_end, x_start:x_end, 2]
                valid_depths = depth_window[valid_mask[y_start:y_end, x_start:x_end]]

                # Calculate median depth if valid depths exist
                if len(valid_depths) > 0:
                    depth = np.median(valid_depths) / 1000.0  # Convert to meters
                else:
                    depth = -1  # Invalid depth

                # Store detection results
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2),
                    'depth': depth,
                    'center': (center_x, center_y)
                })

                # Draw on image
                cv2.rectangle(left_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if depth > 0:
                    label = f'{class_name} {confidence:.2f} {depth:.2f}m'
                else:
                    label = f'{class_name} {confidence:.2f} (invalid depth)'
                cv2.putText(left_rect, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return left_rect, disparity, detected_objects

    def run(self):
        try:
            while True:
                # Capture frames
                ret_left, left_frame = self.left_cam.read()
                ret_right, right_frame = self.right_cam.read()

                if not ret_left or not ret_right:
                    print("Failed to capture frames")
                    break

                # Process frames
                result_frame, disparity, detections = self.detect_objects_with_depth(
                    left_frame, right_frame)

                # Normalize disparity for visualization
                disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
                disparity_vis = disparity_norm.astype(np.uint8)
                disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

                # Display results
                cv2.imshow('Detection + Depth', result_frame)
                cv2.imshow('Disparity', disparity_color)

                # Print detections
                for det in detections:
                    print(f"Detected {det['class']} at {det['depth']:.2f}m")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.left_cam.release()
            self.right_cam.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify your calibration file path
    calib_file = 'C:/My files/academic/DTU/xtra research/deeplabcut/robotfish_data/Robotfish/stereo_calibration.npz'

    # Create detector instance
    detector = StereoObjectDetector(
        calib_file=calib_file,
        left_index=0,
        right_index=1
    )
    detector.run()
```
