import tensorflow as tf
import keras
import os

latent_dim = 1024

K = tf.keras.backend


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


inputs = tf.keras.layers.Input(shape=[128, 128, 3])
z = tf.keras.layers.Conv2D(8, strides=2, kernel_size=3, padding="same", activation="selu")(inputs)
z = tf.keras.layers.Conv2D(16, strides=2, kernel_size=3, padding="same", activation="selu")(z)
z = tf.keras.layers.Conv2D(32, strides=2, kernel_size=3, padding="same", activation="selu")(z)
z = tf.keras.layers.Conv2D(64, strides=2, kernel_size=3, padding="same", activation="selu")(z)
z = tf.keras.layers.Conv2D(128, strides=2, kernel_size=3, padding="same", activation="selu")(z)
z = tf.keras.layers.Conv2D(256, strides=2, kernel_size=3, padding="same", activation="selu")(z)

z = tf.keras.layers.Flatten()(z)
codings_mean = tf.keras.layers.Dense(latent_dim)(z)  # μ
codings_log_var = tf.keras.layers.Dense(latent_dim)(z)  # γ
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

variational_encoder.summary()

