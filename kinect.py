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