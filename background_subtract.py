import numpy as np
import cv2 as cv2
RESOLUTION = 24

def pixelate(img):
    height, width = img.shape[:2]
    w, h = (RESOLUTION, RESOLUTION)

    #Resize input to pixelated size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    #Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output

cap = cv2.VideoCapture(0)
#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv.bgsegm.BackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
#fgbg = cv.createBackgroundSubtractorKNN(detectShadows=True)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    pixelated_image = pixelate(fgmask)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG MASK Frame', fgmask)
    cv2.imshow('pixelated', pixelated_image)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cap.release()
cv2.destroyAllWindows()