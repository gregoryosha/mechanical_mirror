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
    for i in range(24):
        for j in range(8):
            if (img[j][i] == 0):
                ang = OUT_ANG
            else:
                ang = IN_ANG
            box_address = int(i/4) + (6 * int(j/4))
            servo_arr(pca_arr[box_address].channels[3 - i%4 + 4*(j%4)]).angle = ang


def main():
    global FRAME_COUNT
    # i2c = busio.I2C(board.SCL, board.SDA)
    i2c = busio.I2C()
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
                    data = ser.read_until()
                    data = data.decode("utf-8","ignore")
                    # print(data)
                    rows = data[:-1].strip().split(';')
                    # Split each row into elements using spaces as delimiters
                    img = [list(map(int, row.split())) for row in rows]

                    display(img, servo_arr, pca_arr)
                    print(f"frame count: {FRAME_COUNT}")
                    print(f"Buffer size: {ser.in_waiting}")
                    FRAME_COUNT += 1

if __name__ == '__main__':
    main()
