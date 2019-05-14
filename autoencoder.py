import tifffile
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import os
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

height = 1024
width = 1024
channels = 3
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape = (None, height,width,channels)) # 1024*1024*3
conv1 = tf.layers.conv2d(X,filters=32,kernel_size=8, strides=[4,4],padding="SAME",activation=tf.nn.relu) # 256*256*32
conv2 = tf.layers.conv2d(conv1, filters=32,kernel_size=8, strides=[2,2],padding="SAME",activation=tf.nn.relu) # 128*128*32
conv3 = tf.layers.conv2d(conv2, filters=32,kernel_size=4,strides=[2,2],padding="SAME",activation=tf.nn.relu) # 64*64*32
conv4 = tf.layers.conv2d(conv3, filters=1, kernel_size=2,strides=[2,2],padding="SAME",activation=tf.nn.relu) # 32*32*1 latent space
deconv4 = tf.layers.conv2d_transpose(conv4, filters=1, kernel_size=2,strides=[2,2],padding="SAME",activation=tf.nn.relu) # 64*64*32
deconv3 = tf.layers.conv2d_transpose(deconv4, filters=32,kernel_size=4,strides=[2,2],padding="SAME",activation=tf.nn.relu) # 128*128*32
deconv2 = tf.layers.conv2d_transpose(deconv3, filters=32,kernel_size=8, strides=[2,2],padding="SAME",activation=tf.nn.relu) # 256*256*32
deconv1 = tf.layers.conv2d_transpose(deconv2,filters=3,kernel_size=8, strides=[4,4],padding="SAME",activation=tf.nn.relu) # 1024*1024*3

reconstruction_loss = tf.reduce_mean(tf.square(deconv1 - X))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 20
data_directory = "test-RGB"
def trainNet():
    with tf.Session() as sess:
        init.run()
        for iteration in range(len(os.listdir(data_directory))//batch_size):
            batch = get_image_batch(data_directory, iteration*batch_size, batch_size)
            sess.run(training_op, feed_dict={X: batch})
            print(deconv1.shape)
            print(loss)
        save_path = saver.save(sess, "./savedModels/autoencoder1.ckpt")


def get_image_batch(path,index,amount):
    files = os.listdir(path)
    batch = list()
    logging_hook = tf.train.LoggingTensorHook(tensors={"X"},every_n_iter=5)
    for i in range(index, index+amount):
        batch.append(tifffile.imread(path + "/"+files[i]))
    return np.array(batch)

#trainNet()

def try_net():
    with tf.Session() as sess:
        saver.restore(sess, "./savedModels/autoencoder1.ckpt")
        img = tifffile.imread("./Track1/JAX_218_003_RGB.tif")
        images= [img]
        res = deconv1.eval(feed_dict={X: images})
        print(conv1)
        plt.imshow(res[0])
        plt.show()

try_net()