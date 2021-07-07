import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json

template = cv.imread('template.jpg',0) # read as grayscale
img_filled = cv.imread('test_filled.jpg',0) # read as grayscale

with open("template.json", 'r') as jsonFile:
    rois = json.load(jsonFile)

img1 = template
img2 = img_filled

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
distances = []
points_1 = []
points_2 = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        distances.append(m.distance)
        
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt

        points_1.append(pt1)
        points_2.append(pt2)

# get the 4 best matches:
good = [[x,y,z] for _, x, y, z in sorted(zip(distances, good, points_1, points_2), key=lambda data: data[0])][0:4]

coordinates = [x[1:3] for x in good]

""" For evaluating the matches

# cv.drawMatchesKnn expects list of lists as matches.
#img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,[x[0] for x in good],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3),plt.show()

img1_toshow = img1.copy()
img2_toshow = img2.copy()

for coordinate in coordinates:
    cv.circle(img1_toshow, (int(coordinate[0][0]), int(coordinate[0][1])), 20, (255,0,0))
    cv.circle(img2_toshow, (int(coordinate[1][0]), int(coordinate[1][1])), 20, (255,0,0))

cv.namedWindow('image1', cv.WINDOW_NORMAL)
cv.imshow("image1", img1_toshow)
cv.namedWindow('image2', cv.WINDOW_NORMAL)
cv.imshow("image2", img2_toshow)
cv.waitKey(0)
"""

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

coordinates_1 = np.array([x[0] for x in coordinates]) # template
coordinates_2 = np.array([x[1] for x in coordinates]) # image

h, status = cv.findHomography(coordinates_1, coordinates_2)

img2_toshow = img2.copy()

for roi in rois:
    points = np.array([(*x, 1) for x in roi[0]])

    width = roi[1]
    height = roi[2]

    points_transformed = h.dot(points.T).T

    for point in points_transformed:
        cv.circle(img2_toshow, (int(point[0]), int(point[1])), 2, (255,0,0), 10)

cv.namedWindow('image2', cv.WINDOW_NORMAL)
cv.imshow("image2", img2_toshow)
cv.waitKey(0)