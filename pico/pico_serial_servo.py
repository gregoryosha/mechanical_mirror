"""
Listens to the REPL port.
Receives color information and displays it on the NEOPIXEL.
Receives blink command and blinks once.
Sends button press and release.


This uses the optional second serial port available in Circuitpython 7.x
Activate it in the boot.py file with the following code

import usb_cdc
usb_cdc.enable(console=True, data=True)

Some boards might require disabling USB endpoints to enable the data port.
"""

import board
import usb_cdc
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo


################################################################
usb_cdc.data.timeout = 0.1

#Servo Global variables
ROW_INDEX = 0 # Change for each pico 0-5
BOX_NUM = 6 # constant for picos addressing a single row
IN_ANG = 80
OUT_ANG = 120
FRAME_COUNT = 0 #Used for debugging 

def display(img, servo_arr, pca_arr) -> None:
    for n in range(24*8):
        j = n//24 #height pixel
        i = n%24 #width pixel
        if (img[n] == 0):
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

################################################################
# loop-y-loop
################################################################    global FRAME_COUNT
i2c = busio.I2C(board.GP5, board.GP4) #i2c = busio.I2C(board.SCL, board.SDA) for raspi
# i2c = busio.I2C()
print("Starting serial connection... ")

pca_arr = []
for n in range(BOX_NUM):
    pca_arr.append(PCA9685(i2c, address= (0x40 + n + ROW_INDEX*6)))
    pca_arr[n].frequency = 50
servo_arr = servo.Servo
print("Servo shields initialized... ")

while True:
    if usb_cdc.data.in_waiting > 0:
        data = usb_cdc.data.read(size=72) #data is stored in on/off => 72 bytes
        img = decodeStates(data)
        display(img, servo_arr, pca_arr)
        # print(f"frame count: {FRAME_COUNT}")
        # print(f"Buffer size: {ser.in_waiting}")
        FRAME_COUNT += 1
