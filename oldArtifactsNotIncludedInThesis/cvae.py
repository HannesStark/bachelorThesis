from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import errno
from tifffile import imread

from IPython import display


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
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
                tf.keras.layers.Dense(units=32 * 32 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(32, 32, 32)),
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
        return self.decode(eps, apply_sigmoid=True)

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


optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    reconstruction_loss = -tf.reduce_sum(tf.abs(x_logit - x))
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(reconstruction_loss + logpz - logqz_x)


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    filename = 'generatedImages/image_at_epoch_{:04d}.png'.format(epoch)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(filename)


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
    batch /= 255.
    return batch


epochs = 10
latent_dim = 1024
num_examples_to_generate = 16
batch_size = 100
test_train_ratio = 1 / 8  # is only used if test_size is null
test_size = 200  # if test_size is null the test_train_ratio will be used
data_source_dir = "Track1-RGB/Track1-RGB256x256"

random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

generate_and_save_images(model, 0, random_vector_for_generation)

file_paths = get_image_paths(data_source_dir)
np.random.shuffle(file_paths)
n_files = len(file_paths)

if (test_size != None and n_files <= test_size):
    print("error: test_size is larger than the amount of files in the training directory")

split_index = int(n_files // test_train_ratio) if test_size == None else n_files - test_size
training_paths, test_paths = file_paths[:split_index], file_paths[split_index:]
print("number of training images: " + str(len(training_paths)))
print("number of test images: " + str(len(test_paths)))
train_iterations = len(training_paths) // batch_size
test_iterations = len(test_paths) // batch_size

log_file = "log/{}.log".format(time.strftime("%d.%m.%Y %H:%M:%S"))

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for iteration in range(train_iterations):
        first_index = iteration * batch_size
        batch = get_image_batch(training_paths, first_index, batch_size)
        gradients, loss = compute_gradients(model, batch)
        apply_gradients(optimizer, gradients, model.trainable_variables)
        print("Epoch: {}/{}...".format(epoch, epochs),
              "Iteration: {}/{}...".format(iteration + 1, train_iterations),
              "Images: {}/{}...".format((iteration + 1) * batch_size, train_iterations * batch_size),
              "Training loss: {:.4f}".format(loss))
        f = open(log_file, "a")
        f.write("Epoch: {}/{}...".format(epoch, epochs) +
                "Iteration: {}/{}...".format(iteration + 1, train_iterations) +
                "Images: {}/{}...".format((iteration + 1) * batch_size, train_iterations * batch_size) +
                "Training loss: {:.4f}".format(loss) + "\n")
        f.close()
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for iteration in range(test_iterations):
            first_index = iteration * batch_size
            batch = get_image_batch(test_paths, first_index, batch_size)
            loss(compute_loss(model, batch))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
        generate_and_save_images(
            model, epoch, random_vector_for_generation)

saveLocation = 'saved/v3_256x256RGB_to{}.h5'.format(latent_dim)
if not os.path.exists(os.path.dirname(saveLocation)):
    try:
        os.makedirs(os.path.dirname(saveLocation))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
model.save_weights(saveLocation)


def display_image(epoch_no):
    return PIL.Image.open('generatedImages/image_at_epoch_{:04d}.png'.format(epoch_no))


plt.imshow(display_image(epochs))
plt.axis('off')  # Display images

anim_file = 'generatedImages/cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('generatedImages/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython

if IPython.version_info >= (6, 2, 0, ''):
    display.Image(filename=anim_file)
