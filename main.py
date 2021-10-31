from slam import run_slam
import ekf
import os
import numpy as np
import cv2

# Create necessary arrays
PCL = None
Cameras = np.array([[0, 0, 0]])
Image_Library = []



for image in os.listdir("Images"):
    frame = cv2.imread("Images/"+str(image))
    PCL, Cameras, Image_Library = run_slam(frame, PCL, Cameras, Image_Library)

Cameras = np.transpose(Cameras)
Final_Point_Cloud = np.asarray(PCL.points)
Final_Point_Cloud = np.transpose(Final_Point_Cloud)

# EKF
ekf.run_ekf(np.transpose(Cameras), np.transpose(Final_Point_Cloud))


