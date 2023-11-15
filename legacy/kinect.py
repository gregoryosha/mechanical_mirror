
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
import freenect2 as fn2

# We use numpy to process the raw IR frame
import numpy as np

import cv2






def get_next_frames(device):
    # These are the types of the frames we want to capture and the order
    # they should be returned.
    required_types = [fn2.FrameType.Color, fn2.FrameType.Depth, fn2.FrameType.Ir]
    
    # Store incoming frame in this dictionary keyed by type.
    frames_by_type = {}
    
    for frame_type, frame in device:
        # Record frame
        frames_by_type[frame_type] = frame
        
        # Try to return a frame for each type. If we get a KeyError, we need to keep capturing
        try:
            return [frames_by_type[t] for t in required_types]
        except KeyError:
            pass # This is OK, capture the next frame

if __name__ == "__main__":
    # Open default device
    device = fn2.Device()

    # Start the device
    with device.running():
        while True:
            color, depth, ir = get_next_frames(device)
            cv2.imshow('color image', color.to_array()) 
            cv2.imshow('depth image', depth.to_array())
            cv2.imshow('IR image', ir.to_array())
            if cv2.waitKey(10) == 27: 
                break
        
    cv2.destroyAllWindows() 