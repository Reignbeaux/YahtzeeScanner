import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import math

grid_height = 13
grid_filled_width = 4 # how many players did participate / are filled?
grid_full_width = 6 # how wide is the grid actually?

# Load the image
image = cv.imread("filled_2.jpg", cv.IMREAD_GRAYSCALE)

# edge detection:
dst = cv.Canny(image, 100, 320, None, 3)

cv.imshow("Canny", dst)

# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst) # keep a copy of the original

# Extract lines out of the canny transformed image 
lines_hough = cv.HoughLines(dst, 1, np.pi / 180, 250, None, 0, 0)

lines_info = []

# extract parameters of the lines and draw them
if lines_hough is not None:
    for i in range(0, len(lines_hough)):
        rho = lines_hough[i][0][0]
        theta = lines_hough[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho

        pt1 = np.array([int(x0 + 1000*(-b)), int(y0 + 1000*(a))])
        pt2 = np.array([int(x0 - 1000*(-b)), int(y0 - 1000*(a))])
        cv.line(cdst, pt1, pt2, (0,0,255), 1, cv.LINE_AA)

        angle = np.arctan2(*(pt2 - pt1))
        lines_info.append([i, pt1, pt2, angle])

cv.imshow("Source", image)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv.waitKey()

lines_info = np.array(lines_info)
angles = 360/(2*np.pi)*lines_info[:,3]

#plt.hist(angles, density=True, bins=100)
#plt.show()

# Filter the angles: Only keep the two biggest peaks in the histogram that are roughly 90° appart

current_indices_A = []
current_indices_B = []

tollerance = 2

for i, angle_A in enumerate(angles):
    angle_B = angle_A + 90

    new_indices_A = np.where((angles > (angle_A - tollerance)) & (angles < (angle_A + tollerance)))
    new_indices_B = np.where((angles > (angle_B - tollerance)) & (angles < (angle_B + tollerance)))

    if (len(new_indices_A) + len(new_indices_B)) > (len(current_indices_A) + len(current_indices_B)):
        current_indices_A = new_indices_A
        current_indices_B = new_indices_B

to_keep = np.append(current_indices_A, current_indices_B)
angles = angles[to_keep]
lines_info = lines_info[to_keep]

#print(angles)

#plt.hist(angles, density=True, bins=100)
#plt.show()

# TODO: 
# - draw the image with the filtered lines only
# - remove outliers from the lines, keep only the actual grid
# - for each possible rectangle: decide if it is empty or not
# - 

"""
# First, apply a threshold to get binary images
template = cv.adaptiveThreshold(template,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,51,15)
img_filled = cv.adaptiveThreshold(img_filled,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,51,15)
# TODO: Calculate reasonable values for the thresholding from input image dimensions

plt.imshow(template, cmap=plt.cm.gray)
plt.figure()
plt.imshow(img_filled, cmap=plt.cm.gray)
plt.show()
"""