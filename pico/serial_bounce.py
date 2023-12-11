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
import digitalio
import time
import usb_cdc

################################################################
# init board's LEDs for visual output
# replace with your own pins and stuff
################################################################
# Pin object for controlling onboard LED
led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT
################################################################
# prepare values for the loop
################################################################

usb_cdc.data.timeout = 0.1

################################################################
# loop-y-loop
################################################################

while True:
    # read the secondary serial line by line when there's data
    if usb_cdc.data.in_waiting > 0:
        data = usb_cdc.data.readline().decode()
        data = "Recieved: " + data + "\n"
        
    # send the data out once everything to be sent is gathered
        usb_cdc.data.write(data.encode())

    time.sleep(0.1)