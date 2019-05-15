import tensorflow as tf
from tifffile import imread
from os import listdir
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

img = imread("./Track1/JAX_218_003_RGB.tif")

tf.logging.set_verbosity(tf.logging.INFO)
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

height = 1024
width = 1024
channels = 3
learning_rate = 0.001

X = tf.placeholder(tf.float32, shape = (None, height,width,channels)) # 1024*1024*3
conv1 = tf.layers.conv2d(X,filters=32,kernel_size=(12,12), strides=(4,4) ,padding="SAME",activation=tf.nn.relu) # 256*256*32
conv2 = tf.layers.conv2d(conv1, filters=32,kernel_size=(12,12), strides=(2,2),padding="SAME",activation=tf.nn.relu) # 128*128*32
conv3 = tf.layers.conv2d(conv2, filters=32,kernel_size=(12,12),strides=(2,2),padding="SAME",activation=tf.nn.relu) # 64*64*32
latent_mean = tf.layers.conv2d(conv3, filters=32, kernel_size=(12,12),strides=(2,2),padding="SAME",activation=None) # 32*32*32
latent_gamma = tf.layers.conv2d(conv3, filters=32, kernel_size=(12,12),strides=(2,2),padding="SAME",activation=None) # 32*32*32
noise = tf.random_normal(tf.shape(latent_gamma), dtype=tf.float32)
latent_space = latent_mean + tf.exp(0.5 * latent_gamma) * noise # 32*32*32
deconv4 = tf.layers.conv2d_transpose(latent_space, filters=32, kernel_size=(12,12),strides=(2,2),padding="SAME",activation=tf.nn.relu) # 64*64*32
deconv3 = tf.layers.conv2d_transpose(deconv4, filters=32,kernel_size=(12,12),strides=(2,2),padding="SAME",activation=tf.nn.relu) # 128*128*32
deconv2 = tf.layers.conv2d_transpose(deconv3, filters=32,kernel_size=(12,12), strides=(2,2),padding="SAME",activation=tf.nn.relu) # 256*256*32
deconv1 = tf.layers.conv2d_transpose(deconv2,filters=3,kernel_size=(12,12), strides=(4,4),padding="SAME",activation=tf.nn.relu) # 1024*1024*3

reconstruction_loss = tf.reduce_mean(tf.square(deconv1 - X))
#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
latent_loss = 0.5* tf.reduce_sum(tf.exp(latent_gamma)+tf.square(latent_mean)-1 - latent_gamma)

#loss = tf.add_n([reconstruction_loss] + reg_losses)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 1
batch_size = 10
data_directory = "test-RGB"

def train_net():
    with tf.Session() as sess:
        init.run()
        iterations = len(listdir(data_directory))//batch_size
        for epoch in range(n_epochs):
            for iteration in range(iterations):
                batch = get_image_batch(data_directory, iteration*batch_size, batch_size)
                batch_loss, batch_training_op, batch_deconv1 = sess.run([loss, training_op, deconv1], feed_dict={X: batch})
                print("Epoch: {}/{}...".format(epoch + 1, n_epochs),"Iteration: {}/{}...".format(iteration + 1, iterations),"Training loss: {:.4f}".format(batch_loss))
                print(batch_training_op)
                print(batch_deconv1)
        save_path = saver.save(sess, "./savedModels/autoencoder3.ckpt")


def get_image_batch(path,index,amount):
    files = listdir(path)
    batch = list()
    for i in range(index, index+amount):
        batch.append(imread(path + "/"+files[i]))
    return np.array(batch)


def test_net():
    with tf.Session() as sess:
        saver.restore(sess, "./savedModels/autoencoder3.ckpt")
        images= [img]
        res = deconv1.eval(feed_dict={X: images})
        print(res)
        plt.imshow(res[0])
        plt.show()

train_net()
test_net()

print("X" + str(X.shape))
print("conv1" + str(conv1.shape))
print("conv2" + str(conv2.shape))
print("conv3" + str(conv3.shape))
print("latent" + str(latent_space.shape))
print("deconv4" + str(deconv4.shape))
print("deconv3" + str(deconv3.shape))
print("deconv2" + str(deconv2.shape))
print("deconv1" + str(deconv1.shape))

