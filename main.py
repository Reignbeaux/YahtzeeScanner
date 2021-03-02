import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

img_empty = cv.imread('test_empty.jpg',0) # read as grayscale
edges = cv.Canny(img_empty,100,140)
plt.imshow(edges,cmap = 'gray')
plt.title('Empty image'), plt.xticks([]), plt.yticks([])

plt.figure()

img_filled = cv.imread('test_filled.jpg',0) # read as grayscale
edges = cv.Canny(img_filled,100,140)
plt.imshow(edges,cmap = 'gray')
plt.title('Filled image'), plt.xticks([]), plt.yticks([])

plt.show()

# Feature matching
img2 = img_filled
img1 = img_empty

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
for m,n in matches:
    if m.distance < 0.4*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

plt.figure()

plt.imshow(edges,cmap='gray')

roi=cv.selectROI(edges, fromCenter = False)

# Set rectangle