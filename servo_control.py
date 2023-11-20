#!/usr/bin/env python
import time
import serial
import json

ser = serial.Serial(
        port='/dev/serial0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        timeout=0.1
)
while True:
        if ser.in_waiting > 0:
                line = ser.readline()
                line = line.decode("utf-8","ignore")
                img = json.loads(line)
                print(img[0,0])
