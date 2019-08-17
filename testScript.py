import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

cls = cv2.imread("Track1-Truth/JAX_004_006_CLS.tif")

print(cls.shape)

for i in range(0, len(cls)):
    for j in range(0, len(cls[0])):
        if 9 in cls[i, j]:
            cls[i, j] = [255, 255, 255]
plt.imshow(cls)
plt.show()


#vegetation 5
# building 6
#ground 2
# water 9
# clutter 65