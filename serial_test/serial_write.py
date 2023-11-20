#!/usr/bin/env python
import time
import serial

ser = serial.Serial(
        port='/dev/ttyUSB0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        timeout=0.1
)
def write_read(x): 
        ser.write(bytes(x, 'utf-8')) 
        line = ser.readline() 
        line = line.decode("utf-8","ignore")
        return line
while True: 
        num = input("Enter a number: ") # Taking input from user 
        value = write_read(num) 
        print(value) # printing the value 
