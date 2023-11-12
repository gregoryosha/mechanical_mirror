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
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.bgsegm.BackgroundSubtractorGMG()
#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    THRESHHOLD = 150
    thresh, b_w_image = cv2.threshold(fgmask, THRESHHOLD, 255, cv2.THRESH_BINARY)
    b_w_image = cv2.cvtColor(b_w_image, cv2.COLOR_GRAY2RGB)
    pixelated_image = pixelate(b_w_image)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG MASK Frame', fgmask)
    cv2.imshow('pixelated', pixelated_image)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cap.release()
cv2.destroyAllWindows()