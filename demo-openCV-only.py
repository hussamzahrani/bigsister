from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

project_id = "the-big-sister"
model_id = "chest_v20180802215908"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
ap.add_argument("-v", "--thermal", required=True, help="path to thermal images directory")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
for imagePath, thermal_path in zip(paths.list_images(args["images"]), paths.list_images(args["thermal"])):
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))

    thermal = cv2.imread(thermal_path)
    thermal = imutils.resize(thermal, width=min(400, image.shape[1]))

    orig = image.copy()
    rows, cols, channels = image.shape
    blank_image = np.zeros((rows, cols), dtype=np.uint8)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(blank_image, (xA, yA), (xB, yB), (255, 255, 255), -1)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 3)

    res2 = cv2.bitwise_and(thermal, thermal, mask=blank_image)
    hsv = cv2.cvtColor(res2, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 70, 70])
    upper_blue = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(res2, res2, mask=mask)

    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))

    # show the output images
    cv2.imshow("masking thermal image of detected piligrim", res2)
    cv2.imshow("Detecting abnormal temperature on detected piligrim", res)
    cv2.imshow("Piligramage detected", image)
    cv2.waitKey(0)




