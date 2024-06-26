
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
import RPi.GPIO as GPIO

#Pins
RESET_PIN = 4

servo_arr = servo.Servo

in_ang = 80
out_ang = 140

short_sleep = 0.5
long_sleep = 3

BOX_NUM = 36

pca_arr = []
for n in range(BOX_NUM):
    pca_arr.append(PCA9685(i2c, address= (0x40 + n)))
for n in range(BOX_NUM):
    pca_arr[n].frequency = 50
servo_arr = servo.Servo

GPIO.setup(RESET_PIN, GPIO.OUT)  # type: ignore
GPIO.output(RESET_PIN, False)  # type: ignore


while True:
    print("Input servo angle: ")
    cmd = input()
    if (cmd == 'wave'):
        while True:
            try:
                for n in range(BOX_NUM):
                    for j in range(4):
                        for i in range(4):
                            servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = out_ang
                        time.sleep(0.1)
                for n in range(BOX_NUM):
                    for j in range(4):
                        for i in range(4):
                            servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = in_ang
                        time.sleep(0.1)
            except:
                for n in range(BOX_NUM):
                    for j in range(4):
                        for i in range(4):
                            servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = in_ang
                        time.sleep(0.1)
                for n in range(BOX_NUM):
                    for j in range(4):
                        for i in range(4):
                            servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = None
                            time.sleep(0.001)
                break
    else:
        for n in range(BOX_NUM):
            for j in range(4):
                for i in range(4):
                    servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = int(cmd)
                    time.sleep(0.02)
        
        for n in range(BOX_NUM):
            for j in range(4):
                for i in range(4):
                    servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = None
                    time.sleep(0.005)
        
        GPIO.output(RESET_PIN, True)
        time.sleep(1)
        GPIO.output(RESET_PIN, False)
