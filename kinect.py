
"""
import freenect2 as freenect
import cv2 
import numpy as np 
 
while True: 
    depth, timestamp = freenect.sync_get_depth() 
    np.clip(depth, 0, 2**10 - 1, depth) 
    depth >>= 2 
    depth = depth.astype(np.uint8) 
    blur = cv2.GaussianBlur(depth, (5, 5), 0) 
    cv2.imshow('image', blur) 
    if cv2.waitKey(10) == 27: 
        break 
 
cv2.destroyAllWindows() 
"""
"""
Simple IR camera using freenect2. Saves captured IR image
to output.jpg.

"""
# Import parts of freenect2 we're going to use
from freenect2 import Device, FrameType

# We use numpy to process the raw IR frame
import numpy as np

import cv2

# Open default device
device = Device()

# Start the device
with device.running():
    # For each received frame...
    for type_, frame in device:
        # ...stop only when we get an IR frame
        if type_ is FrameType.Ir:
            break

# Outside of the 'with' block, the device has been stopped again

# The received IR frame is in the range 0 -> 65535. Normalise the
# range to 0 -> 1 and take square root as a simple form of gamma
# correction.
ir_image = frame.to_array()
ir_image /= ir_image.max()
ir_image = np.sqrt(ir_image)

cv2.imshow('image', 256*ir_image) 
if cv2.waitKey(10) == 27: 
    cv2.destroyAllWindows() 

