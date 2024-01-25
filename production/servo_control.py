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

invert = False

PAUSE_TIME = time.time()
TIME_TILL_RESET = 5
paused = True

def display(img) -> None:
    global BOX_NUM, PREV_IMG, PAUSE_TIME
    global paused, invert

    change_count = 0
    if ((time.time() - PAUSE_TIME) > TIME_TILL_RESET and (not paused)):
        reload()
        paused = True

    for n in range(16 * BOX_NUM):
        if (img[n] != PREV_IMG[n]):
            #Checking for a pause
            change_count += 1
            paused = False

            j = n//24 #height pixel
            i = n%24 #width pixel
            if (img[n] == 0):
                if (invert):
                    ang = IN_ANG
                else:
                    ang = OUT_ANG
            else:
                if (invert):
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
                    reload()
                    print(f"overloaded, ValueError: {report}")
    PREV_IMG = img    

    if (change_count != 0):
        PAUSE_TIME = time.time()
        

def decodeStates(data: bytes) -> list[int]:
    out_states = []
    for byte in data:
        for i in range(8):
            bit = (byte >> (7 - i)) & 0b1
            out_states.append(bit)
    return out_states

def reload():
    global invert 
    ser.write(bytes('pause', 'utf-8')) 
    print("reloading...")
    time.sleep(0.5)
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
        reload()
    except ValueError as report:
        print(f"overloaded, ValueError: {report}")
        reload()
        
    invert = not invert
    print("finished reload")
    if (ser.in_waiting >= 288):
        print(f"Buffer size: {ser.in_waiting}")
        blank = ser.read(288)



        
def control_servos():
    global FRAME_COUNT, PAUSE_TIME
    global pca_arr, servo_arr
    
    print("Starting serial connection... ")
    pca_arr = []
    try:
        for n in range(BOX_NUM):
            pca_arr.append(PCA9685(i2c, address= (0x40 + n + ROW_INDEX*6)))
            pca_arr[n].frequency = 50
            time.sleep(0.02)
        servo_arr = servo.Servo
        print("Servo shields initialized... ")
    
        while True:
            if ser.in_waiting > 0:
                    data = ser.read(size=72) #data is stored in on/off => 72 bytes
                    img = decodeStates(data)
                    display(img)

                    if (ser.in_waiting >= 288):
                        print(f"Buffer size: {ser.in_waiting}")
                        blank = ser.read(288)

    except KeyboardInterrupt:
        print("Exiting and reseting servos...")
        reload()
        exit
    except OSError as report:
        print(f"OSError in Main: {report}")
        control_servos()
        
def main():
    global i2c, ser
    i2c = busio.I2C(board.SCL, board.SDA) #i2c = busio.I2C(board.SCL, board.SDA) for raspi
    # i2c = busio.I2C()
    ser = serial.Serial(
            port='/dev/serial0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
            baudrate = 115200,
            timeout=1
    )
    time.sleep(3)
    ser.write(bytes('start', 'utf-8')) 
    print("Starting Mirror!")
    control_servos()
                
if __name__ == '__main__':
    main()
