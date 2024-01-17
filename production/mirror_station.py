import argparse
import sys

import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import serial

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Servo Global variables
SER_TIME = time.time()
FRAME_TIME = 0.5
PREV_IMG = [0] * 576


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None
RESOLUTION = 24

def send_to_pi(img, ser):
    global SER_TIME
    global FRAME_TIME
    global PREV_IMG
    
    if (time.time() - SER_TIME) > FRAME_TIME:
        #Resize input to pixelated size
        w, h = (RESOLUTION, RESOLUTION)
        pix_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img_list = pix_img[:, :, 0].flatten().tolist()

        change_count = 0
        for pix, n in enumerate(img_list):
            if (pix != PREV_IMG[n]):
                change_count += 1

        FRAME_TIME = change_count * 0.7 / 576 + 0.1

        # Join the elements into a single string
        ser.write(encodeStates(img_list))    
        ser.flush() 
        # print(f"Waiting: {ser.out_waiting}")
        PREV_IMG = img_list
        SER_TIME = time.time()

def encodeStates(states: list[int]) -> bytes:
    out_bytes = b""

    # iterate over indices 0, 8, 16,...
    for chunk_idx in range(0, len(states), 8):
        new_byte = 0b0

        # iterate over eight states, starting at chunk_idx
        for state in states[chunk_idx : chunk_idx + 8]:
            # insert bit by shifting new_byte over by a bit
            # and inserting the new bit
            bit = state > 0
            new_byte = (new_byte << 1) | bit
        out_bytes += new_byte.to_bytes(1, 'big')

    return out_bytes

def run_mirror(model:str='pose_landmarker.task', num_poses: int=1,
        min_pose_detection_confidence: float=0.5,
        min_pose_presence_confidence: float=0.5, min_tracking_confidence: float=0.5,
        camera_id: int=0, width: int=1280, height: int=960) -> None:
    """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the pose landmarker model bundle.
      num_poses: Max number of poses that can be detected by the landmarker.
      min_pose_detection_confidence: The minimum confidence score for pose
        detection to be considered successful.
      min_pose_presence_confidence: The minimum confidence score of pose
        presence score in the pose landmark detection.
      min_tracking_confidence: The minimum confidence score for the pose
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """
    ser = serial.Serial(
        port='/dev/ttyUSB0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        timeout=1
    )

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 48)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)

    # Visualization parameters
    mask_color = (0, 0, 0)  # white
    bg_color = (255, 255, 255)

    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global DETECTION_RESULT, COUNTER

        DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the pose landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=True,
        result_callback=save_result)
    
    detector = vision.PoseLandmarker.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run pose landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Set the background image
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = bg_color
        current_frame = bg_image
    
        if DETECTION_RESULT:
            # Draw landmarks.
            if DETECTION_RESULT.segmentation_masks is not None:
                segmentation_mask = DETECTION_RESULT.segmentation_masks[0].numpy_view()
                mask_image = np.zeros(image.shape, dtype=np.uint8)
                mask_image[:] = mask_color
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1

                visualized_mask = np.where(condition, mask_image, bg_image)
                current_frame = visualized_mask
        send_to_pi(current_frame, ser)

    detector.close()
    cap.release()


if __name__ == '__main__':
    run_mirror()
