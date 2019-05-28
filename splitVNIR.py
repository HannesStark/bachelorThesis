from tifffile import imread
from matplotlib import pyplot as plt
import numpy as np

img = imread("somepath")
images = list()
for i in range(0,1024):
    for j in range(0,1024):
