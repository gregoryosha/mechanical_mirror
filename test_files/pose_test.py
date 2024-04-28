import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

RESOLUTION = 24

def pixelate(img):
    height, width = img.shape[:2]
    w, h = (RESOLUTION, RESOLUTION)

    #Resize input to pixelated size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    # print(temp[0,0,0])


    #Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    enable_segmentation=True,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    try:
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1   
    except:
        condition = np.zeros(image.shape, dtype=np.uint8)
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    pose_image = np.zeros(image.shape, dtype=np.uint8)

    # THRESHHOLD = 150
    # grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Convert image to greyscale
    # thresh, b_w_image = cv2.threshold(grey_image, THRESHHOLD, 255, cv2.THRESH_BINARY)
    # b_w_image = cv2.cvtColor(b_w_image, cv2.COLOR_GRAY2RGB)

    BG_COLOR = (255, 255, 255) #white screen
    POSE_COLOR = (0, 0, 0) #black fill
    bg_image[:] = BG_COLOR
    pose_image[:] = POSE_COLOR

    annotated_image = np.where(condition, pose_image, bg_image)
    pixelated_image = pixelate(annotated_image)
    # Draw the pose annotation on the image.
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Pixelated View', cv2.flip(pixelated_image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()