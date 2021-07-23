import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import math

grid_height = 13
grid_filled_width = 4 # how many players did participate / are filled?
grid_full_width = 6 # how wide is the grid actually?

# Load the image
image = cv.imread("filled.jpg", cv.IMREAD_GRAYSCALE)

# edge detection:
dst = cv.Canny(image, 100, 320, None, 3)

#cv.imshow("Canny", dst)

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
        #cv.line(cdst, pt1, pt2, (0,0,255), 1, cv.LINE_AA)

        angle = np.arctan2(*(pt2 - pt1))
        lines_info.append([i, pt1, pt2, angle])

#cv.imshow("Source", image)
#cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
#cv.waitKey()

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
angle_A_average = np.average(angles[current_indices_A])

print(angle_A_average)

angle_B_average = np.average(angles[current_indices_B])
angles = angles[to_keep]

lines_info_A = lines_info[current_indices_A]
lines_info_B = lines_info[current_indices_B]
lines_info = lines_info[to_keep]

for line in lines_info:
    cv.line(cdst, line[1], line[2], (0,0,255), 1, cv.LINE_AA)

def get_distances(angle_orthogonal, lines_info):

    mid_point = np.array([cdst.shape[0] / 2, cdst.shape[1] / 2])

    # TODO: This does not appear to be the mid point!

    cv.circle(cdst, mid_point.astype(int), 5, (255,0,0), 5)

    second_point = mid_point + np.array([1*np.tan(angle_orthogonal*2*np.pi/360), 1])

    cv.line(cdst, (mid_point-1000*(second_point-mid_point)).astype(int), (mid_point+1000*(second_point-mid_point)).astype(int), (255,0,0), 5, cv.LINE_AA)

    ds = []

    for line in lines_info:
        # find intersection

        line_mid = line[1]
        line_vec = np.array(line[2]) - np.array(line[1])

        # Solve linear system of two equations:
        # orth = mid_point + t * (second_point - mid_point) = line_mid + t2 * line_vec

        # =>

        t = (line_mid[0]*line_vec[1] - line_mid[1]*line_vec[0] + line_vec[0]*mid_point[1] - line_vec[1]*mid_point[0])/(line_vec[0]*mid_point[1] - line_vec[0]*second_point[1] - line_vec[1]*mid_point[0] + line_vec[1]*second_point[0])
        ds.append(t)

    return ds

ds_B = get_distances(angle_A_average, lines_info_B)
ds_A = get_distances(angle_B_average, lines_info_A)

#cv.imshow("Source", image)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
#cv.waitKey()

plt.hist(ds_A, density=True, bins=500)
plt.figure()
plt.hist(ds_B, density=True, bins=500)
plt.show()

# TODO: 
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