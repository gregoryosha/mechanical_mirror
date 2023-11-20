#!/usr/bin/env python
import time
import serial
import json
import numpy as np

ser = serial.Serial(
        port='/dev/serial0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        timeout=0.1
)
while True:
        if ser.in_waiting > 0:
                data = ser.readline()
                data = data.decode("utf-8","ignore")
                img = np.matrix(data)
                print(img[0,0])
