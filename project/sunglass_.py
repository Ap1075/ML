import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)
#imread_color = 1
#imread_unchanged = -1
##cv2.startWindowThread()
##cv2.namedWindow("preview")
cv2.imshow('preview',img)
cv2.waitKet(0)
cv2.destroyAllWindows
