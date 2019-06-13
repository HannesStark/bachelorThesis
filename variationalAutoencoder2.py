from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
from tifffile import imread


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(1024, 1024, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=64 * 64 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(64, 64, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    print("X: " + str(x))
    print("logits: " + str(x_logit))
    reconstruction_loss = tf.reduce_mean(tf.math.abs(x_logit - x))
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    latent_loss = tf.reduce_mean(tf.math.abs(logpz - logqz_x))
    print("latent Loss: " + str(latent_loss))
    print("reconstruction loss: " + str(reconstruction_loss))
    return reconstruction_loss


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


def get_image_paths(dir):
    image_paths = os.listdir(dir)
    data = list()
    for i in image_paths:
        data.append(dir + "/" + i)
    return data


def get_image_batch(files, index, batchsize):
    batch = list()
    for i in range(index, index + batchsize):
        batch.append(imread(files[i]))
    batch = np.array(batch, dtype=np.float32)
    batch /= 127.5
    batch -= 1.
    return batch


def train_net(model, n_epochs, batchsize, files):
    optimizer = tf.keras.optimizers.Adam(0.001)
    np.random.shuffle(files)
    n_files = len(files)
    iterations = n_files // batchsize
    for epoch in range(n_epochs):
        for iteration in range(iterations):
            batch = get_image_batch(files, iteration, batchsize)
            batch_gradients, batch_loss = compute_gradients(model, batch)
            optimizer.apply_gradients(zip(batch_gradients, model.trainable_variables))
            print("Epoch: {}/{}...".format(epoch + 1, n_epochs),
                  "Iteration: {}/{}...".format(iteration + 1, iterations),
                  "Images: {}/{}...".format((iteration + 1) * batchsize, iterations * batchsize),
                  "Training loss: {:.4f}".format(batch_loss))
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        # random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
        # generate_and_save_images(model, epoch, random_vector_for_generation)


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))
    predictions += 1.
    predictions *= 127.
    predictions = predictions.astype()
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0])
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def try_net(model, image):
    image /= 127.5
    image -= 1.
    mean, logvar = model.encode([image])
    encoding = model.reparameterize(mean, logvar)
    erg = model.decode(encoding)
    print("erg: " + str(erg))
    erg += 1.
    erg *= 127.5
    return np.array(erg, dtype=np.int)


epochs = 1
batch_size = 3
latent_dim = 64
num_examples_to_generate = 16
vaemodel = CVAE(latent_dim)
checkpoint_path = './training_checkpoints/cp.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)

file_paths = get_image_paths("test-RGB")
batch = list()
for i in range(len(file_paths)):
    batch.append(imread(file_paths[i]))
batch = np.array(batch, dtype=np.float32)
batch /= 127.5
batch -= 1.
vaemodel.fit(batch)
# train_net(vaemodel, epochs, batch_size, file_paths)
vaemodel.save_weights('./savedModels/path_to_my_model.h5')
#vaemodel.load_weights('./savedModels/path_to_my_model.h5')
# new_model = tf.keras.models.load_('./savedModels/variationalAutoencoder2.h5')
tryimg = imread("./Track1-RGB/JAX_467_013_RGB.tif")
tryimg = np.array(tryimg, dtype=np.float32)
result = try_net(vaemodel, np.array([tryimg]))
print(result)
plt.imshow(result[0])
plt.show()
