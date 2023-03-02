#Created by MediaPipe
#Modified by Augmented Startups 2021
#Zoom Virtual Background in OpenCV Python
#Watch Computer Vision Tutorial at www.augmentedstartups.info/VisionStore
import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

RESOLUTION = 24

def pixelate(img):
    height, width = img.shape[:2]
    w, h = (RESOLUTION, RESOLUTION)

    #Resize input to pixelated size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    #Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output


# For webcam input:
BG_COLOR = (255, 255, 255) # green screen
cap = cv2.VideoCapture(0)
prevTime = 0
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
  bg_image = None
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #print(image.shape)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = selfie_segmentation.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.1
    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    #bg_image = cv2.imread('backgrounds/1.png')
    #bg_image = cv2.GaussianBlur(image, (55, 55), 0)
    if bg_image is None:
      bg_image = np.zeros(image.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR
    blank_fill = np.zeros(image.shape, dtype=np.uint8)
    blank_fill[:] = (0,0,0)
    output_image = np.where(condition, blank_fill, bg_image)
    #Get FrameRate
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    output_image = pixelate(output_image)
    cv2.putText(output_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('Filtered Screen', output_image)
    cv2.imshow('Camera View', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
