from tifffile import imread
from matplotlib import pyplot as plt
import numpy as np
import exifread
import pprint as pp
import zipfile
import os

def splitVNIR():
    img = imread("./Track1-MSI-3/OMA_315_008_MSI.tif")
    images = np.zeros(shape=(8, 1024, 1024, 1),dtype=np.int)
    for i in range(0, 1024):
        for j in range(0, 1024):
            for channel in range(0, 8):
                images[channel,i,j,0]=img[i,j,channel]

    print(images[0].shape)
    f, axarr = plt.subplots(4,2)
    axarr[0,0].imshow(images[0], cmap="Greys")
    axarr[1,0].imshow(images[1], cmap="Greys")
    axarr[2,0].imshow(images[2], cmap="Greys")
    axarr[3,0].imshow(images[3], cmap="Greys")
    axarr[0,1].imshow(images[4], cmap="Greys")
    axarr[1,1].imshow(images[5], cmap="Greys")
    axarr[2,1].imshow(images[6], cmap="Greys")
    axarr[3,1].imshow(images[7], cmap="Greys")
    plt.show()

def readEXIF():
    image = open("./Track1-Truth/JAX_004_016_CLS.tif", 'rb')
    exifTags = exifread.process_file(image)
    print(type(exifTags))
    pp.pprint(exifTags)

def readCls():
    img = imread("./Track1-Truth/JAX_004_016_CLS.tif")
    print(img)
    print(type(img))
    print(np.max(img))
    plt.imshow(img)
    plt.show()

def readAndDisplayDSM():
    imread()

def testUnzip():
    os.listdir(os.getcwd())
    zip_ref = zipfile.ZipFile("test-RGB.zip", 'r')
    zip_ref.extractall("./")
    zip_ref.close()

testUnzip()