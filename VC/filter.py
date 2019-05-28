import numpy as np
from matplotlib import pyplot as plt

import cv2

img = cv2.imread("./pictures/bridge/bridge_004.jpg")

gauss_img = np.array(img)
pseudo_gauss_img = gauss_img

pseudo_gauss_kernel = np.array([[1, 1, 1],
                                [1, 4, 1],
                                [1, 1, 1]])
gauss_kernel = np.array([[1, 1, 1],
                         [1, 4, 1],
                         [1, 1, 1]])

pseudo_gauss_img = cv2.filter2D(src=img, dst=pseudo_gauss_img, ddepth=-1, kernel=pseudo_gauss_kernel)
gauss_img= cv2.filter2D(src=img, dst=gauss_img, ddepth=-1, kernel=gauss_kernel)
plt.imshow(gauss_img)
plt.show()
plt.imshow(pseudo_gauss_img)
plt.show()
