""" This script allow you to generate a new template for scanning new types of sheets.
You need a photo of an empty sheet for this. The template data will be written to a json file.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json

#input_file = input("Enter filename of input image for the template.\n")
input_file = "template.jpg"
#output_file = input("Enter output filename of the template (*.json).\n")
output_file = "template.json"

img = cv.imread(input_file, 0) # read as grayscale

# Show the image to the user and subscribe to mouse events

mousePositions = []

def mouse_event(event,x,y,flags,param):
    global mousePositions
    if event == cv.EVENT_LBUTTONDOWN:
        mousePositions.append((x,y))

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.setMouseCallback('image', mouse_event)
cv.imshow('image', img)

print("Please select one or multiple ROIs in the template. For this, click the four edges of the ROI.\n"
"If you clicked the four edges, simply press any key.\n"
"For each ROI, you have to then specify the number of columns and rows.\n"
"If you don't want to add a new ROI, simply press a key without selecting any edges.\n\n")

rois = []

while True:

    cv.waitKey(0)

    if len(mousePositions) != 4:
        print("Selecting ROI finished, since number of added points wasn't 4.\n")
        break

    columns = int(input("Please enter the number of columns of the ROI.\n"))
    rows = int(input("Please enter the number of rows of the ROI.\n"))

    rois.append((mousePositions, columns, rows))

    mousePositions = []

    print("Please select the next ROI or press any key to finish.\n")

# Check if all rois have the same number of columns

first = rois[0][1]
for roi in rois[1:]:
    if roi[1] != first:
        print("Number of columns must be the same on all ROIs.")
        exit(-1)

print("Final configuration of template:\n")
print(rois)

# Write ROI information to json file

with open(output_file, 'w') as jsonFile:
    json.dump(rois, jsonFile)