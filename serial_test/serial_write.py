#!/usr/bin/env python
import time
import serial

ser = serial.Serial(
        port='/dev/ttyUSB0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)
def write_read(x): 
        ser.write(bytes(x, 'utf-8')) 
        time.sleep(0.05) 
        data = ser.readline() 
        return data 
while True: 
        num = input("Enter a number: ") # Taking input from user 
        value = write_read(num) 
        print(value) # printing the value 
