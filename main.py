import atexit
from unicodedata import name
import cv2
import numpy as np

RESOLUTION = 32

def pixelate(img):
    height, width = img.shape[:2]
    w, h = (RESOLUTION, RESOLUTION)

    #Resize input to pixelated size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    #Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        
        success, frame = cap.read()
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Convert image to greyscale
        thresh, frame_black = cv2.threshold(frame_grey, 100, 255, cv2.THRESH_BINARY)

        cv2.imshow("Frame", frame)
        cv2.imshow("Pixelation", pixelate(frame_black))

        key = cv2.waitKey(1) #esc key
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()