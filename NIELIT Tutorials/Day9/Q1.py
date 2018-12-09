import cv2 as cv
import numpy as np


img = np.zeros((512,512,3), np.uint8)

img[1:50,1:50] = (0,128,0)

#print(img)

cv.imshow('a',img)
cv.waitKey(0)
