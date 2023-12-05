#!/usr/bin/env python
import time
import serial

while True:
        try:
                print("try serial port")
                port = input()
                ser = serial.Serial(
                        port=f'/dev/{port}', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
                        baudrate = 115200,
                        timeout=0.1
                )
                while True:
                        if ser.in_waiting > 0:
                                line = ser.readline()
                                line = line.decode("utf-8","ignore")
                                print(line)
        except:
                print("port failed")

