from tifffile import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf


def split_images(directory, split_by=2, resize=None):
    image_names = os.listdir(directory)
    image_paths = list()
    for i in image_names:
        image_paths.append(directory + "/" + i)
    img1 = imread(image_paths[0])
    original_height = len(img1)
    original_width = len(img1[0])
    height = original_height // split_by
    width = original_width // split_by
    dirname = directory + str(height) + "x" + str(width) + "split" + str(split_by)
    if not resize == None:
        dirname = directory + str(resize[0]) + "x" + str(resize[1]) + "split" + str(split_by)
    print(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
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
                    counter) + "split" + str(split_by) + "." + filetype

                destination_image = img[i * height:(i + 1) * height, j * width:(j + 1) * width]
                if not resize == None:
                    destination_image = cv2.resize(destination_image, dsize=resize, interpolation=cv2.INTER_CUBIC)
                    filepath = dirname + "/" + image_name_without_filetype + "_" + str(resize[0]) + "x" + str(
                        resize[1]) + "_" + str(
                        counter) + "split" + str(split_by) + "." + filetype
                print(filepath)
                imsave(filepath, destination_image)


def plot_normal_distribution():
    with tf.compat.v1.Session().as_default():
        codings = tf.random.normal(shape=[1024]).eval()
        array = np.array(codings)
        print(array.shape)
        plt.plot(array, 'o')
        plt.show()


split_images("test-RGB", 8)
