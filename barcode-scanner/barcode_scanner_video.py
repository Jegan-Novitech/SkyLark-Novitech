from imutils.video import VideoStream
from pyzbar import pyzbar
import argparse
import datetime
import imutils
import time
import cv2

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = VideoStream(src=0,usePiCamera=True).start()
#vs=cv2.VideoCapture(1)
time.sleep(2.0)
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    frame =cv2.rectangle(frame,(384,0),(710,228),(0,255,0),3)
    crop_img = frame[0:228, 384:710]

    barcodes = pyzbar.decode(crop_img)

    # loop over the detected barcodes
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw
        # the bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        # draw the barcode data and barcode type on the image
        text = "{} ".format(barcodeData)

    cv2.imshow("Barcode Scanner", frame)
    cv2.imshow("Barcode Scanner1", crop_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] cleaning up...")
csv.close()
cv2.destroyAllWindows()
vs.stop()