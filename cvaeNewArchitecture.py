from tensorflow import keras
import tensorflow as tf
from keras import losses
import numpy as np
import os
import tifffile as tif
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


K = keras.backend

codings_size = 1024


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean



inputs = keras.layers.Input(shape=[128, 128, 3])
z = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="selu")(inputs)
z = keras.layers.MaxPool2D(pool_size=2)(z)
z = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu")(z)
z = keras.layers.MaxPool2D(pool_size=2)(z)
z = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu")(z)
z = keras.layers.MaxPool2D(pool_size=2)(z)
z = keras.layers.Flatten()(z)
codings_mean = keras.layers.Dense(codings_size)(z)  # μ
codings_log_var = keras.layers.Dense(codings_size)(z)  # γ
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])



decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(16*16*64)(decoder_inputs)
x = keras.layers.Reshape([16,16,64])(x)
x = keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="selu")(x)
x = keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="selu")(x)
outputs = keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid")(x)
variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[outputs])


_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.Model(inputs=[inputs], outputs=[reconstructions])


latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)
variational_ae.add_loss(K.mean(latent_loss) / 196608.)
variational_ae.compile(loss=losses.mean_absolute_error, optimizer="rmsprop", metrics=['accuracy'])

batch_size = 128
train_dir = "test-RGB128x128"
validation_dir = "validation"

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'test-RGB256x256',
        batch_size=128)

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        batch_size=128)

ceil(num_samples / batch_size)

history = variational_ae.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
#variational_ae.load_weights("modelNewArchitecure.h5")

variational_ae.save_weights("modelNewArchitecure.h5")

pred_on = X_valid[:10]

predictions = variational_ae.predict(pred_on,batch_size=1)

for i in range(0, len(pred_on)):
    plt.imshow(pred_on[i])
    plt.savefig("img" + str(i))

for i in range(0, len(predictions)):
    plt.imshow(predictions[i])
    plt.savefig("pred" + str(i))

codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()
print(images)
print(images.shape)

for i in range(0, len(images)):
    plt.imshow(images[i])
    plt.savefig("img" + str(i))