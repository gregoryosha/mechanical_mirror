
# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import board
import time
import busio
import digitalio

# Import the PCA9685 module.
from adafruit_pca9685 import PCA9685

#Servo Juice
from adafruit_motor import servo

i2c = busio.I2C(board.SCL, board.SDA)


servo_arr = servo.Servo

in_ang = 80
out_ang = 120

short_sleep = 0.5
long_sleep = 3

box_num = 6

pca_arr = [PCA9685(i2c, address=0x40), PCA9685(i2c, address=0x41), PCA9685(i2c, address=0x42), PCA9685(i2c, address=0x43), PCA9685(i2c, address=0x44), PCA9685(i2c, address=0x45)]

for n in range(box_num):
    pca_arr[n].frequency = 50

while True:
    print("Input servo angle: ")
    cmd = input()
    if (cmd == 'wave'):
        while True:
            try:
                for n in range(box_num):
                    for j in range(4):
                        for i in range(4):
                            servo_arr(pca_arr[n].channels[i*4 + j]).angle = out_ang
                        time.sleep(0.1)
                for n in range(box_num):
                    for j in range(4):
                        for i in range(4):
                            servo_arr(pca_arr[n].channels[i*4 + j]).angle = in_ang
                        time.sleep(0.1)
            except:
                for n in range(box_num):
                    for j in range(4):
                        for i in range(4):
                            servo_arr(pca_arr[n].channels[i*4 + j]).angle = in_ang
                        time.sleep(0.1)
                break
    else:
        for i in range(16):
            for n in range(box_num):
                servo_arr(pca_arr[n].channels[i]).angle = int(cmd)
                time.sleep(0.0001)
