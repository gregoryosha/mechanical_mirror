# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run pose landmarker."""

import argparse
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None
RESOLUTION = 24

def pixelate(img):
    w, h = (RESOLUTION, RESOLUTION)

    #Resize input to pixelated size
    output = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    print(output[0,0])
    return output

def run(model: str, num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
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

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    mask_color = (0, 0, 0)  # white
    bg_color = (255, 255, 255)

    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

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

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = bg_color
        current_frame = bg_image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)
        if DETECTION_RESULT:
            # Draw landmarks.
            if DETECTION_RESULT.segmentation_masks is not None:
                segmentation_mask = DETECTION_RESULT.segmentation_masks[0].numpy_view()
                mask_image = np.zeros(image.shape, dtype=np.uint8)
                mask_image[:] = mask_color
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1

                visualized_mask = np.where(condition, mask_image, bg_image)
                current_frame = visualized_mask
        pixelate(current_frame)
        cv2.imshow('pose_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the pose landmarker model bundle.',
        required=False,
        default='pose_landmarker.task')
    parser.add_argument(
        '--numPoses',
        help='Max number of poses that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minPoseDetectionConfidence',
        help='The minimum confidence score for pose detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minPosePresenceConfidence',
        help='The minimum confidence score of pose presence score in the pose '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the pose tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    args = parser.parse_args()

    run(args.model, int(args.numPoses), args.minPoseDetectionConfidence,
        args.minPosePresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()