import tensorflow as tf
from tifffile import imread
from os import listdir
import numpy as np
from matplotlib import pyplot as plt

def get_image_batch(path, index, amount):
    files = listdir(path)
    batch = list()
    for i in range(index, index + amount):
        batch.append(imread(path + "/" + files[i]))
    batch = np.array(batch)
    np.random.shuffle(batch)
    return batch


def print_shapes():
    print("inputs_     " + str(inputs_.shape))
    print("conv1       " + str(conv1.shape))
    print("maxpool1    " + str(maxpool1.shape))
    print("conv2       " + str(conv2.shape))
    print("maxpool2    " + str(maxpool2.shape))
    print("conv3       " + str(conv3.shape))
    print("latent_space" + str(latent_space.shape))
    print("upsample1   " + str(upsample1.shape))
    print("conv4       " + str(conv4.shape))
    print("upsample2   " + str(upsample2.shape))
    print("conv5       " + str(conv5.shape))
    print("upsample3   " + str(upsample3.shape))
    print("conv6       " + str(conv6.shape))
    print("logits      " + str(logits.shape))

learning_rate = 0.001

inputs_ = tf.placeholder(tf.float32, (None, 1024, 1024, 3), name='inputs')
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
latent_space = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
upsample1 = tf.image.resize_images(latent_space, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
upsample2 = tf.image.resize_images(conv4, size=(512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
upsample3 = tf.image.resize_images(conv5, size=(1024, 1024), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3, 3), padding='same', activation=None)
reconstruction_loss = tf.reduce_mean(tf.square(logits - inputs_))
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
latent_loss = 0.5* tf.reduce_sum(tf.exp(latent_gamma)+tf.square(latent_mean)-1 - latent_gamma)

#loss = tf.add_n([reconstruction_loss] + reg_losses)
loss = reconstruction_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()

n_epochs = 1
batch_size = 5
data_directory = "test-RGB"


def train_net():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            n_images = len(listdir(data_directory))
            iterations = n_images // batch_size
            for iteration in range(iterations):
                first_index = iteration * batch_size
                batch = get_image_batch(data_directory, first_index, batch_size)
                batch_loss, batch_training_opt = sess.run([loss, training_op], feed_dict={inputs_: batch})
                print("Epoch: {}/{}...".format(epoch + 1, n_epochs),
                      "Iteration: {}/{}...".format(iteration + 1, iterations),
                      "Images: {}/{}...".format((iteration + 1) * batch_size, iterations * batch_size),
                      "Training loss: {:.4f}".format(batch_loss))

        save_path = saver.save(sess, "./savedModels/autoencoder2.ckpt")


def try_net():
    img = imread("./Track1/JAX_218_003_RGB.tif")
    images = [img]
    with tf.Session() as sess:
        saver.restore(sess, "./savedModels/autoencoder2.ckpt")
        res = logits.eval(feed_dict={inputs_: images})
        result_img = np.array(res[0], dtype=int)
        print("Maximum value of image: " + str(np.max(result_img)))
        f, img_array = plt.subplots(1, 2)
        img_array[0].imshow(img)
        img_array[1].imshow(result_img)
        plt.show()





print_shapes()
#train_net()
try_net()
