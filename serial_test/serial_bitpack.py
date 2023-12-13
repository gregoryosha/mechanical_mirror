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

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
SER_TIME = time.time()
FRAME = 0
DETECTION_RESULT = None
RESOLUTION = 24

def send_to_pi(ser):
    global SER_TIME
    global FRAME
    if (time.time() - SER_TIME) > 0.05:
        pix_img = np.random.rand(24,24,3)
        pix_img[0,0,0] = 0
        pix_img[1,4,0] = FRAME%2
        img_list = pix_img[:, :, 0].flatten().tolist()

        # Join the elements into a single string
        ser.write(encodeStates(img_list))    
        ser.flush()
        SER_TIME = time.time()
        print(f"frame: {FRAME}")
        FRAME += 1

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
        out_bytes += new_byte.to_bytes(1,'big')

    return out_bytes


def main():
    ser = serial.Serial(
        port='/dev/ttyACM2', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        timeout=1
    )
    while True:
        send_to_pi(ser)
        


if __name__ == '__main__':
    main()
