import tensorflow as tf
from tifffile import imread
from os import listdir
import numpy as np
from matplotlib import pyplot as plt


def get_image_batch(path, index, amount):
    files = listdir(path)
    batch = list()
    logging_hook = tf.train.LoggingTensorHook(tensors={"X"}, every_n_iter=5)
    for i in range(index, index + amount):
        batch.append(imread(path + "/" + files[i]))
    return np.array(batch)


learning_rate = 0.001
inputs_ = tf.placeholder(tf.float32, (None, 1024, 1024, 3), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')
### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 14x14x32
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 7x7x32
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 4x4x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 7x7x16
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
upsample2 = tf.image.resize_images(conv4, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 14x14x16
conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
upsample3 = tf.image.resize_images(conv5, size=(1024, 1024), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 28x28x32
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
logits = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=(3, 3), padding='same', activation=None)
# Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
reconstruction_loss = tf.reduce_mean(tf.square(logits - inputs_))
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# latent_loss = 0.5* tf.reduce_sum(tf.exp(latent_gamma)+tf.square(latent_mean)-1 - latent_gamma)

# loss = tf.add_n([reconstruction_loss] + reg_losses)
loss = reconstruction_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()

img = imread("./Track1/JAX_218_003_RGB.tif")
images = [img]

n_epochs = 1
batch_size = 5
data_directory = "test-RGB"


def train_net():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(n_epochs):
            for iteration in range(len(listdir(data_directory)) // batch_size):
                batch = get_image_batch(data_directory, iteration * batch_size, batch_size)
                batch_cost, batch_training_opt = sess.run([loss, training_op], feed_dict={inputs_: batch})
                print("Epoch: {}/{}...".format(e + 1, n_epochs), "Training loss: {:.4f}".format(batch_cost))

        save_path = saver.save(sess, "./savedModels/autoencoderAnderer.ckpt")
        res = logits.eval(feed_dict={inputs_: images})
        plt.imshow(res[0])
        plt.show()


def try_net():
    with tf.Session() as sess:
        saver.restore(sess, "./savedModels/autoencoderAnderer.ckpt")
        res = logits.eval(feed_dict={inputs_: images})
        result_img = np.array(res[0], dtype=int)
        print(np.max(result_img))
        plt.imshow(img)
        plt.show()
        plt.imshow(result_img)
        plt.show()


print("conv3" + str(conv3.shape))
print("encoded" + str(encoded.shape))
print("upsample1" + str(upsample1.shape))
print("conv4" + str(conv4.shape))
print("upsample2" + str(upsample2.shape))
print("conv4" + str(conv5.shape))
