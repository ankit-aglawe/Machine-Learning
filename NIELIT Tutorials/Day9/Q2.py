import numpy as np
import cv2 as cv

img = np.zeros((512,512,3),np.uint8)


cv.line(img, (10,1),(500,100),(0,0,128),3)

cv.imshow('b', img)
cv.waitKey(0)

cv.line(img,(1,100),(500,10),(0,128,0),3)

cv.imshow('d', img)
cv.waitKey(0)


cv.rectangle(img,(10,1),(500,100),(128,0,0),5)

cv.imshow('c', img)
cv.waitKey(0)
