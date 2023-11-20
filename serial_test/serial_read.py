#!/usr/bin/env python
import time
import serial

ser = serial.Serial(
        port='/dev/ttyUSB0',
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)

while True:
        if ser.in_waiting > 0:
                line = ser.readline()
                line = line.decode("utf-8","ignore")
                print(line)