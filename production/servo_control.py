#!/usr/bin/env python
import serial

import board
import time
import busio
import digitalio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo


#Servo Global variables
ROW_INDEX = 0 # Change for each pico 0-5
BOX_NUM = 36 # constant for picos addressing a single row
IN_ANG = 80
OUT_ANG = 140
FRAME_COUNT = 0 #Used for debugging 
PREV_IMG = [0] * 576

PAUSE_TIME = time.time()
TIME_TILL_RESET = 3


def display(img, servo_arr, pca_arr) -> None:
    global BOX_NUM
    global PREV_IMG
    if (len(img) == 576):
        for n in range(16 * BOX_NUM):
            if (img[n] != PREV_IMG[n]):
                j = n//24 #height pixel
                i = n%24 #width pixel
                if (img[n] == 0):
                    ang = OUT_ANG
                else:
                    ang = IN_ANG
                try:
                    box_address = int(i/4) + (6 * int(j/4))
                    servo_arr(pca_arr[box_address].channels[3 - i%4 + 4*(j%4)]).angle = ang
                    time.sleep(0.001)
                except OSError as report:
                     print(f"OSError: {report}")
                except ValueError as report:
                     print(f"overloaded, ValueError: {report}")
        PREV_IMG = img
    else:
        print("img size unequal...")
        print(f"image size: {len(img)}")

def decodeStates(data: bytes) -> list[int]:
    out_states = []
    for byte in data:
        for i in range(8):
            bit = (byte >> (7 - i)) & 0b1
            out_states.append(bit)
    return out_states

def reload(servo_arr, pca_arr):
    print("reloading...")
    try: 
        for n in range(BOX_NUM):
            for j in range(4):
                for i in range(4):
                    servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = None
                    time.sleep(0.001)
                    servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = IN_ANG
                    time.sleep(0.001)
        
        for n in range(BOX_NUM):
            for j in range(4):
                for i in range(4):
                    servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = None
                time.sleep(0.001)

    except OSError as report:
        print(f"OSError: {report}")
    except ValueError as report:
        print(f"overloaded, ValueError: {report}")
        
    print("finished reload")
    if (ser.in_waiting >= 288):
        print(f"Buffer size: {ser.in_waiting}")
        blank = ser.read(288)



                            

        
def main():
    global FRAME_COUNT, PAUSE_TIME
    global ser
    i2c = busio.I2C(board.SCL, board.SDA) #i2c = busio.I2C(board.SCL, board.SDA) for raspi
    # i2c = busio.I2C()
    ser = serial.Serial(
            port='/dev/serial0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
            baudrate = 115200,
            timeout=1
    )
    print("Starting serial connection... ")
    pca_arr = []
    try:
        for n in range(BOX_NUM):
            pca_arr.append(PCA9685(i2c, address= (0x40 + n + ROW_INDEX*6)))
            pca_arr[n].frequency = 50
            time.sleep(0.02)
        servo_arr = servo.Servo
        print("Servo shields initialized... ")

        paused = False
        PAUSE_TIME = time.time()
    
        while True:
            if ((time.time() - PAUSE_TIME) > TIME_TILL_RESET and (not paused)):
                reload(servo_arr, pca_arr)
                paused = True
            if ser.in_waiting > 0:
                    data = ser.read(size=72) #data is stored in on/off => 72 bytes
                    img = decodeStates(data)
                    display(img, servo_arr, pca_arr)
                    if (ser.in_waiting >= 288):
                        print(f"Buffer size: {ser.in_waiting}")
                        blank = ser.read(288)
                    FRAME_COUNT += 1

                    PAUSE_TIME = time.time()
                    paused = False
    except KeyboardInterrupt:
        print("Exiting and reseting servos...")
        reload(servo_arr, pca_arr)
    except OSError as report:
        print(f"OSError in Main: {report}")
        main()
        

                

if __name__ == '__main__':
    main()
