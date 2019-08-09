from tifffile import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
import exifread
import pprint as pp
import zipfile
import os
import errno


def splitVNIR():
    img = imread("./Track1-MSI-3/OMA_315_008_MSI.tif")
    images = np.zeros(shape=(8, 1024, 1024, 1), dtype=np.int)
    for i in range(0, 1024):
        for j in range(0, 1024):
            for channel in range(0, 8):
                images[channel, i, j, 0] = img[i, j, channel]

    print(images[0].shape)
    f, axarr = plt.subplots(4, 2)
    axarr[0, 0].imshow(images[0], cmap="Greys")
    axarr[1, 0].imshow(images[1], cmap="Greys")
    axarr[2, 0].imshow(images[2], cmap="Greys")
    axarr[3, 0].imshow(images[3], cmap="Greys")
    axarr[0, 1].imshow(images[4], cmap="Greys")
    axarr[1, 1].imshow(images[5], cmap="Greys")
    axarr[2, 1].imshow(images[6], cmap="Greys")
    axarr[3, 1].imshow(images[7], cmap="Greys")
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


def split_images(directory, split_by=2):
    image_names = os.listdir(directory)
    image_paths = list()
    for i in image_names:
        image_paths.append(directory + "/" + i)
    img1 = imread(image_paths[0])
    original_height = len(img1)
    original_width = len(img1[0])
    height = original_height // split_by
    width = original_width // split_by
    dirname = directory + str(height) + "x" + str(width)
    print(dirname)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for iteration in range(len(image_paths)):
        path = image_paths[iteration]
        image_name = image_names[iteration]
        img = imread(path)
        print(path)
        counter = 0
        for i in range(split_by):
            for j in range(split_by):
                counter += 1
                image_name_without_filetype = image_name.split(".", 1)[0]
                filetype = image_name.split(".", 1)[1]
                filepath = dirname + "/" + image_name_without_filetype + "_" + str(height) + "x" + str(
                    width) + "_" + str(
                    counter) + "." + filetype
                print(filepath)
                imsave(filepath, img[i * height:(i + 1) * height, j * width:(j + 1) * width])


#split_images("test-RGB", 8)
