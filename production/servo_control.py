#!/usr/bin/env python
import serial

import board
import time
import busio
import digitalio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

import RPi.GPIO as GPIO


#Pins
RESET_PIN = 4

#Servo Global variables
ROW_INDEX = 0 # Change for each pico 0-5
BOX_NUM = 36 # constant for picos addressing a single row
IN_ANG = 80
OUT_ANG = 140
FRAME_COUNT = 0 #Used for debugging 
PREV_IMG = [0] * 576

inverted = False

PAUSE_TIME = time.time()
TIME_SOFT_RESET = 15
TIME_HARD_RESET = 60 * 15
paused = True
hard_paused = False

def display(img) -> None:
    global BOX_NUM, PREV_IMG, PAUSE_TIME
    global paused, hard_paused

    change_count = 0
    if ((time.time() - PAUSE_TIME) > TIME_SOFT_RESET and (not paused)):
        paused = True
        hard_paused = False
        reload()

    elif ((time.time() - PAUSE_TIME) > TIME_HARD_RESET and (not hard_paused)):
        hard_paused = True
        reload('reset')

    for n in range(16 * BOX_NUM):
        if (img[n] != PREV_IMG[n]):

            #Checking for a pause
            change_count += 1
            paused = False

            j = n//24 #height pixel
            i = n%24 #width pixel
            if (img[n] == 0):
                ang = OUT_ANG
            else:
                ang = IN_ANG
            try:
                box_address = int(i/4) + (6 * int(j/4))
                servo_arr(pca_arr[box_address].channels[3 - i%4 + 4*(j%4)]).angle = ang
                time.sleep(0.0015)
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

def reload(mode: str='null'):
    global inverted, IN_ANG, OUT_ANG
    if (mode == 'null'):
        print('Soft Reset')
        ser.write(bytes('00000', 'utf-8')) #Send 5 bytes for soft reset
    elif (mode == 'reset'):
        print('Long reset')
        ser.write(bytes('000000', 'utf-8')) #Send 6 bytes for hard reset
        # temp = IN_ANG
        # IN_ANG = OUT_ANG
        # OUT_ANG = temp
        # inverted = not inverted

    print('reloading...')
    time.sleep(1.5)

    try: 

        if (mode == 'reset'):
            GPIO.output(RESET_PIN, True)
            time.sleep(10)
            GPIO.output(RESET_PIN, False)
            time.sleep(2)

            for n in range(BOX_NUM):
                for j in range(4):
                    for i in range(4):
                        servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = IN_ANG
                        time.sleep(0.004)

        for n in range(BOX_NUM):
            for j in range(4):
                for i in range(4):
                    servo_arr(pca_arr[n].channels[i*4 + 3-j]).angle = None
                    time.sleep(0.002)
        
        
        
    except OSError as report:
        print(f"OSError: {report}")
        reload()
    except ValueError as report:
        print(f"overloaded, ValueError: {report}")
        reload()
        
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
        reload('reset')
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
    ser.reset_input_buffer()

    GPIO.setup(RESET_PIN, GPIO.OUT)  # type: ignore
    GPIO.output(RESET_PIN, False)  # type: ignore

    while True:
        ser.write(bytes('pi_start', 'utf-8')) 
        print("Waiting on handshake...")
        time.sleep(3)
        if (ser.in_waiting > 0):
            line = ser.readline().decode("utf-8","ignore")
            if (line == 'station_start'):
                print("Computer Handshake! Starting mirror")
                control_servos()
                exit()
                
if __name__ == '__main__':
    main()
