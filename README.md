# SimpleStereoCam

This is a project that i did to improve my understanding of making a stereo camera and calibrate it to estimate distances of objects from the camera.
In the project i used two identical Microsoft modern webcams. I designed a frame for the camera setup using blender and 3D printed it.

## What do you need for the project?
1. Two cameras in the same model to prevent any mismatches in focal length, latency and other external and internal features
2. 3D printer (optional if you need a nice frame for the camera)
3. Calibration board (chess board image printed) - 9x6, 10x7 with known square size measurements
4. Computer with python installed and necessary libraries

## Lets get familiar with camera calibration

You can follow this youtube channel to understand the mathematics behind camera calibration which is highly important to understand the extrinsic and intrinsic parameters of the camera

<a href="https://www.youtube.com/@firstprinciplesofcomputerv3258" style="display: inline-block; background-color: #ff5733; color: white; padding: 10px 20px; text-decoration: none; border-radius: 8px; font-weight: bold;">First principles of Computer vision</a>


[![My YouTube Channel](images/calb.png)](https://www.youtube.com/watch?v=S-UHiFsn-GI&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo)

Checkerboard i used for the calibration - 10x7 squares with 0.25mm size. 
Print this on a A4 paper and double check the measurement of the square before proceeding to calibration!

<img src="images/Checkerboard-A4-25mm-10x7 (1).jpg" alt="Alt text" width="599">

## Getting the first video feed from the camera setup
Here it is important to position your cameras in nearly same horizontal and vertical alignment to prevent any form of calibration errors. 
(Note - There are always inaccuracies when using standard consumer webcameras. but if you use industrial grade cameras with hardware synchronization, you can get a very accurate result)



## Baseline distance between the camera's optical centers

<img src="images/baseline.png" alt="Alt text" width="800">

When choosing the baseline distance between two cameras, it's essential to consider the application environment—whether you're targeting long-range or short-range depth measurements. The baseline must be optimized for your specific use case.

In stereo vision systems, the baseline (the fixed distance between the two cameras) is key to determining depth accuracy. Objects appear in slightly different positions in the two camera images, a difference called **disparity**. This disparity is used to calculate an object's depth, as described by the equation:

Z = (f * B) / d


where:
- **Z** is the depth,
- **f** is the focal length,
- **B** is the baseline distance,
- **d** is the disparity.

Larger baseline:
- Better depth resolution for distant objects
- Wider field of view
- But harder to find corresponding points between images due to larger differences in perspective
- AND larger minimum measurable distance (blind spot close to cameras)

Smaller baseline:
- Better for measuring depth of closer objects
- Easier to find corresponding points (better matching)
- BUT reduced depth resolution for far objects
- AND more sensitive to calibration errors


## Getting the first video feed from the setup

```python
import cv2
import threading
import time
import numpy as np
from datetime import datetime

# Define a class to handle stereo camera operations
class StereoCamera:
    def __init__(self, left_id=0, right_id=1):
        # Initialize the left and right cameras using the provided device IDs
        self.left_cam = cv2.VideoCapture(left_id)
        self.right_cam = cv2.VideoCapture(right_id)

        # Configure camera settings for both cameras:
        # - Resolution: 960x540 pixels
        # - Frame Rate: 30 FPS
        # - Buffer Size: 1 (to minimize delay)
        for cam in [self.left_cam, self.right_cam]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cam.set(cv2.CAP_PROP_FPS, 30)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize FPS tracking variables for each camera
        self.left_fps = 0
        self.right_fps = 0
        self.left_fps_time = time.time()   # Timestamp to track FPS update for left camera
        self.right_fps_time = time.time()  # Timestamp to track FPS update for right camera
        self.frame_count_left = 0          # Frame counter for left camera
        self.frame_count_right = 0         # Frame counter for right camera
        self.fps_update_interval = 1.0     # Interval (in seconds) to update the FPS calculations

        # Warm up cameras by capturing a few frames to let auto-adjustments (like exposure) settle
        print("Warming up cameras...")
        for _ in range(30):
            self.left_cam.read()
            self.right_cam.read()

        # Wait a bit to ensure the cameras are ready
        time.sleep(2)
        print("Cameras ready!")

        # Initialize threading locks for safe access to the camera frames across threads
        self.left_lock = threading.Lock()
        self.right_lock = threading.Lock()

        # Initialize variables to hold the most recent frames from each camera
        self.left_frame = None
        self.right_frame = None

    def update_fps(self, camera='left'):
        # Update the frames per second (FPS) measurement for the specified camera
        if camera == 'left':
            current_time = time.time()
            time_diff = current_time - self.left_fps_time
            # If the specified time interval has passed, update FPS for left camera
            if time_diff >= self.fps_update_interval:
                self.left_fps = self.frame_count_left / time_diff  # Calculate FPS
                self.frame_count_left = 0  # Reset frame count
                self.left_fps_time = current_time  # Reset the time marker
        else:
            current_time = time.time()
            time_diff = current_time - self.right_fps_time
            # If the specified time interval has passed, update FPS for right camera
            if time_diff >= self.fps_update_interval:
                self.right_fps = self.frame_count_right / time_diff  # Calculate FPS
                self.frame_count_right = 0  # Reset frame count
                self.right_fps_time = current_time  # Reset the time marker

    def capture_frames(self):
        # Continuously capture frames from both cameras in a loop
        while True:
            ret_left, left = self.left_cam.read()   # Capture frame from left camera
            ret_right, right = self.right_cam.read()  # Capture frame from right camera

            # If both frames are successfully captured
            if ret_left and ret_right:
                # Use a lock to safely update the left frame and its FPS counter
                with self.left_lock:
                    self.left_frame = left
                    self.frame_count_left += 1
                    self.update_fps('left')

                # Use a lock to safely update the right frame and its FPS counter
                with self.right_lock:
                    self.right_frame = right
                    self.frame_count_right += 1
                    self.update_fps('right')
            else:
                # If capturing a frame failed, print an error message and wait briefly before retrying
                print("Frame capture failed")
                time.sleep(0.1)

    def show_feeds(self):
        # Start a separate thread to capture frames continuously from both cameras
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True  # Mark thread as daemon so it exits when the main program terminates
        capture_thread.start()

        # Create a resizable window for displaying the stereo feeds
        cv2.namedWindow('Stereo Feeds', cv2.WINDOW_NORMAL)

        # Main loop for displaying the camera feeds
        while True:
            # Ensure that frames have been captured from both cameras
            if self.left_frame is not None and self.right_frame is not None:
                # Safely copy the current frames using locks to avoid thread conflicts
                with self.left_lock:
                    left_display = self.left_frame.copy()
                with self.right_lock:
                    right_display = self.right_frame.copy()

                # Get the current timestamp formatted as HH:MM:SS.milliseconds
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]

                # Overlay the timestamp and left camera FPS on the left frame
                cv2.putText(left_display, f"Time: {timestamp}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(left_display, f"Left FPS: {self.left_fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Overlay the timestamp and right camera FPS on the right frame
                cv2.putText(right_display, f"Time: {timestamp}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(right_display, f"Right FPS: {self.right_fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Combine the left and right frames horizontally for side-by-side display
                combined_display = np.hstack((left_display, right_display))

                # Show the combined frame in the window
                cv2.imshow('Stereo Feeds', combined_display)

                # Check if the 'q' key is pressed to exit the display loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clean up: close the window and release camera resources when done
        cv2.destroyAllWindows()
        self.left_cam.release()
        self.right_cam.release()

# Main execution block: run the stereo camera feed if this script is executed directly
if __name__ == "__main__":
    try:
        # Create an instance of the StereoCamera with default device IDs (0 and 1)
        stereo = StereoCamera(left_id=0, right_id=1)
        # Start displaying the stereo camera feeds
        stereo.show_feeds()
    except Exception as e:
        # In case of an error, print the error message and ensure windows are closed
        print(f"Error occurred: {str(e)}")
        cv2.destroyAllWindows()

```
