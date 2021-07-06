import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img_empty = cv.imread('test_empty.jpg',0) # read as grayscale
img_filled = cv.imread('test_filled.jpg',0) # read as grayscale

# Feature matching
img2 = cv.adaptiveThreshold(img_empty,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,5,2)
img1 = cv.adaptiveThreshold(img_filled,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

plt.imshow(img1, cmap=plt.cm.gray)
plt.savefig("test.png")
plt.figure()
plt.imshow(img2, cmap=plt.cm.gray)
plt.show()

orb = cv.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# draw first 50 matches
match_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None)

plt.imshow(match_img)
plt.show()

#########################################################
#plt.figure()

#roi=cv.selectROI(edges, fromCenter = False)

# Set rectangle