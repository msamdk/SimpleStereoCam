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

A larger baseline increases disparity for distant objects, improving measurement accuracy, while a smaller baseline is often better for close-range applications.

