import errno
import tensorflow as tf
from keras import losses
import numpy as np
import os
import tifffile as tiff
from matplotlib import pyplot as plt
import time

K = tf.keras.backend

latent_dim = 1024
batch_size = 128
epochs = 50
num_examples_to_generate = 16
test_train_ratio = 1 / 8  # is only used if test_size is null
test_size = 200  # if test_size is null the test_train_ratio will be used
#data_source_dir = "Track1-RGB/Track1-RGB128x128"


data_source_dir = "test-RGB128x128"


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


inputs = tf.keras.layers.Input(shape=[128, 128, 3])
z = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="selu")(inputs)
z = tf.keras.layers.MaxPool2D(pool_size=2)(z)
z = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu")(z)
z = tf.keras.layers.MaxPool2D(pool_size=2)(z)
z = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu")(z)
z = tf.keras.layers.MaxPool2D(pool_size=2)(z)
z = tf.keras.layers.Flatten()(z)
codings_mean = tf.keras.layers.Dense(latent_dim)(z)  # μ
codings_log_var = tf.keras.layers.Dense(latent_dim)(z)  # γ
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

decoder_inputs = tf.keras.layers.Input(shape=[latent_dim])
x = tf.keras.layers.Dense(16 * 16 * 64)(decoder_inputs)
x = tf.keras.layers.Reshape([16, 16, 64])(x)
x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="selu")(x)
x = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="selu")(x)
outputs = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid")(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)
variational_ae.add_loss(K.mean(latent_loss) / 196608.)
variational_ae.compile(loss=losses.mean_absolute_error, optimizer="rmsprop", metrics=['accuracy'])


def get_image_paths(dir):
    image_paths = os.listdir(dir)
    data = list()
    for i in image_paths:
        data.append(dir + "/" + i)
    return data


def get_image_batch(files, index, batchsize):
    batch = list()
    for i in range(index, index + batchsize):
        batch.append(tiff.imread(files[i]))
    batch = np.array(batch, dtype=np.float32)
    batch /= 255.
    return batch


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

log_file = "{}.log".format(time.strftime("%d_%m_%Y_%H_%M_%S"))

#for epoch in range(1, epochs + 1):
#    for iteration in range(train_iterations):
#        first_index = iteration * batch_size
#        batch = get_image_batch(training_paths, first_index, batch_size)
#        history = variational_ae.train_on_batch(batch, batch)
#        print(history)
#        print("Epoch: {}/{}...".format(epoch, epochs),
#              "Iteration: {}/{}...".format(iteration + 1, train_iterations),
#              "Images: {}/{}...".format((iteration + 1) * batch_size, train_iterations * batch_size))
#        f = open(log_file, "a")
#        f.write(str(history) + "\n" +
#                "Epoch: {}/{}...".format(epoch, epochs) +
#                "Iteration: {}/{}...".format(iteration + 1, train_iterations) +
#                "Images: {}/{}...".format((iteration + 1) * batch_size, train_iterations * batch_size) + "\n")
#        f.close()
#
#    saveLocation = 'saved/v1_128x128RGB_to{}_epoch_{}.h5'.format(latent_dim, epoch)
#    if not os.path.exists(os.path.dirname(saveLocation)):
#        try:
#            os.makedirs(os.path.dirname(saveLocation))
#        except OSError as exc:  # Guard against race condition
#            if exc.errno != errno.EEXIST:
#                raise
#    variational_ae.save_weights(saveLocation)

variational_ae.load_weights("./savedModels/modelNewArchitecureV1.h5")

#variational_ae.save_weights("modelNewArchitecureV1.h5")

pred_on = list()
for path in file_paths[5:15]:
    pred_on.append(tiff.imread(path))
pred_on = np.array(pred_on, dtype=np.float32)
pred_on /= 255.

predictions = variational_ae.predict(pred_on)

for i in range(0, len(pred_on)):
    plt.imshow(pred_on[i])
    plt.savefig("img" + str(i))

for i in range(0, len(predictions)):
    plt.imshow(predictions[i])
    plt.savefig("pred" + str(i))

codings = tf.random.normal(shape=[12, latent_dim])
images_generated = variational_decoder.predict(codings, steps=1)
print(type(images_generated))
print(images_generated.shape)

for i in range(0, len(images_generated)):
    plt.imshow(images_generated[i])
    plt.savefig("generated" + str(i))