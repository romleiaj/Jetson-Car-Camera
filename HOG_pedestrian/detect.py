# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

gst_str = ("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280,"
 "height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! \
        nvvidconv flip-method=0 ! video/x-raw, format=(string)I420 ! \
	videoconvert ! video/x-raw, format=(string)BGR ! \
	appsink")
onboard_cam =  cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

cv2.namedWindow("After NMS", cv2.WINDOW_NORMAL)
cv2.resizeWindow("After NMS", 1920, 1080)

while True:  
    ret_val, image = onboard_cam.read()

# loop over the image paths
# load the image and resize it to (1) reduce detection time
# and (2) improve detection accuracy
    image = imutils.resize(image, width=min(600, image.shape[1]))

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
 	padding=(8, 8), scale=1.05)
 
# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
# show some information on the number of bounding boxes
    #filename = imagePath[imagePath.rfind("/") + 1:]
    #print("[INFO] {}: {} original boxes, {} after suppression".format(
	#filename, len(rects), len(pick)))
 
# show the output images
    #cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
onboard_cam.release()
cv2.destroyAllWindows()
