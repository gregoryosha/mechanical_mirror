#!/usr/bin/env python
import serial
import numpy as np

import board
import time
import busio
import digitalio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo


#Servo Global variables
BOX_NUM = 12
IN_ANG = 80
OUT_ANG = 120
FRAME_COUNT = 0

def display(img, servo_arr, pca_arr):
    for i in range(24*8):
        j = int(i/24)
        if (img[i] == 0):
            ang = OUT_ANG
        else:
            ang = IN_ANG
        box_address = int(i/4) + (6 * int(j/4))
        servo_arr(pca_arr[box_address].channels[3 - i%4 + 4*(j%4)]).angle = ang

def decodeStates(data: bytes) -> list[int]:
    out_states = []
    for byte in data:
        for i in range(8):
            bit = (byte >> (7 - i)) & 0b1
            out_states.append(bit)
    return out_states

def main():
    global FRAME_COUNT
    i2c = busio.I2C(board.SCL, board.SDA)
    # i2c = busio.I2C()
    pca_arr = []
    for n in range(BOX_NUM):
        pca_arr.append(PCA9685(i2c, address= (0x40 + n)))
    for n in range(BOX_NUM):
        pca_arr[n].frequency = 50
    servo_arr = servo.Servo
    ser = serial.Serial(
            port='/dev/serial0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
            baudrate = 115200,
            timeout=1
    )
    print("Starting serial connection... ")
    while True:
            if ser.in_waiting > 0:
                    data = ser.read(size=72)
                    img = decodeStates(data)
                    # print(img)
                    display(img, servo_arr, pca_arr)
                    print(f"frame count: {FRAME_COUNT}")
                    print(f"Buffer size: {ser.in_waiting}")
                    FRAME_COUNT += 1

if __name__ == '__main__':
    main()
