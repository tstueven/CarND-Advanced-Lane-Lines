import cv2
import numpy as np

def hls_thresh(img, channel='s', thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        img_channel = hls[:,:,0]
    elif channel == 'l':
        img_channel = hls[:,:,1]
    elif channel == 's':    
        img_channel = hls[:,:,2]
    binary = np.zeros_like(img_channel)
    binary[(img_channel >= thresh[0]) & (img_channel <= thresh[1])] = 1
    return binary