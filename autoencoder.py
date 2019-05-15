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
conv1 = tf.layers.conv2d(X,filters=8,kernel_size=(8,8), strides=(1,1) ,padding="SAME",activation=tf.nn.relu)
max_pooling2 = tf.layers.max_pooling2d(conv1, pool_size=(4,4), strides=(4,4), padding="SAME")# 256*256*16
conv3 = tf.layers.conv2d(max_pooling2, filters=8,kernel_size=(4,4), strides=(1,1),padding="SAME",activation=tf.nn.relu)
max_pooling3 = tf.layers.max_pooling2d(conv3, pool_size=(4,4), strides=(4,4), padding="SAME")# 256*256*16
conv4 = tf.layers.conv2d(max_pooling3, filters=8,kernel_size=(4,4),strides=(1,1),padding="SAME",activation=tf.nn.relu) # 64*64*32
max_pooling4 = tf.layers.max_pooling2d(conv4, pool_size=(4,4), strides=(2,2), padding="SAME")# 256*256*16
conv5 = tf.layers.conv2d(max_pooling4, filters=8,kernel_size=(4,4),strides=(1,1),padding="SAME",activation=tf.nn.relu) # 64*64*32
flatten6 = tf.layers.flatten(conv5) # 64*64*32
dense7_mean = tf.layers.dense(flatten6, units=4000, activation=tf.nn.relu)
dense7_gamma = tf.layers.dense(flatten6,units=4000, activation=tf.nn.relu)
noise = tf.random_normal(tf.shape(dense7_gamma), dtype=tf.float32)
latent_space = dense7_mean + tf.exp(0.5 * dense7_gamma) * noise
dense8 = tf.layers.dense(latent_space, units=8192, activation=tf.nn.relu)
reshape9 = tf.reshape(dense8, shape=[-1, 32, 32, 8])
upsampler2 = tf.keras.layers.UpSampling2D(size=(2,2))
upsampler4 = tf.keras.layers.UpSampling2D(size=(4,4))
upsample10 = upsampler2.apply(reshape9)
conv11 = tf.layers.conv2d(upsample10, filters=8,kernel_size=(4,4),strides=(1,1),padding="SAME",activation=tf.nn.relu)
upsample12 = upsampler4.apply(conv11)
conv13 = tf.layers.conv2d(upsample12, filters=8,kernel_size=(4,4),strides=(1,1),padding="SAME",activation=tf.nn.relu)
upsample14 = upsampler4.apply(conv13)
conv15 = tf.layers.conv2d(upsample14, filters=3,kernel_size=(4,4),strides=(1,1),padding="SAME",activation=tf.nn.relu)

reconstruction_loss = tf.reduce_mean(tf.square(conv15 - X))
#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
latent_loss = 0.5* tf.reduce_sum(tf.exp(dense7_gamma)+tf.square(dense7_mean)-1 - dense7_gamma)

#loss = tf.add_n([reconstruction_loss] + reg_losses)
loss = reconstruction_loss - latent_loss

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
                batch = get_image_batch_shuffled(data_directory, iteration*batch_size, batch_size)
                batch_loss, batch_training_op, batch_deconv1 = sess.run([loss, training_op, conv15], feed_dict={X: batch})
                print(batch_deconv1)
                print("Epoch: {}/{}...".format(epoch + 1, n_epochs),
                      "Iteration: {}/{}...".format(iteration + 1, iterations),
                      "Training loss: {:.4f}".format(batch_loss))
                print(batch_training_op)
        save_path = saver.save(sess, "./savedModels/autoencoder.ckpt")


def get_image_batch_shuffled(path,index,amount):
    files = listdir(path)
    batch = list()
    for i in range(index, index+amount):
        batch.append(imread(path + "/"+files[i]))
    result = np.array(batch)
    np.random.shuffle(result)
    return result



def try_net():
    with tf.Session() as sess:
        saver.restore(sess, "./savedModels/autoencoder.ckpt")
        images= [img]
        res = conv15.eval(feed_dict={X: images})
        print(res)
        plt.imshow(res[0])
        plt.show()

train_net()
try_net()

print("X" + str(X.shape))
print("conv1" + str(conv1.shape))


