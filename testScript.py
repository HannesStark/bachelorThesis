from tifffile import imread
import numpy as np
import os

dirs = os.listdir("Track1-Truth")
list1 = list()
maxnumasdf = 0
location=""
for i in dirs:
    pic = imread("Track1-Truth\\" + i)
    newmaxas = np.max(pic)
    print(type(maxnumasdf))
    if (newmaxas > maxnumasdf):
        maxnumasdf = newmaxas
        location = i
    print(maxnumasdf)
    print(location)