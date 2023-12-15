#!/usr/bin/env python
import time
import serial

ser = serial.Serial(
        port='/dev/ttyUSB0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        timeout=0.1
)
        
while True: 
        num = input("Enter a number: ") # Taking input from user 
        ser.write(bytes(num, 'utf-8')) 
        if ser.in_waiting:
                line = ser.readline()
                line = line.decode("utf-8","ignore")
                print(line)
        
