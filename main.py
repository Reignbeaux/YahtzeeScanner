import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

template = cv.imread('template.jpg',0) # read as grayscale
img_filled = cv.imread('test_filled.jpg',0) # read as grayscale

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
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        distances.append(m.distance)

print(distances)
good = [x for _,x in sorted(zip(distances, good), key=lambda pair: pair[0])][0:4]

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()