import tensorflow as tf
import keras
import IPython
model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(128 + 128),
            ]
        )



epochs = 10
latent_dim = 128
num_examples_to_generate = 16
batchsize = 100
test_train_ratio = 1 / 8  # is only used if test_size is null
test_size = 200  # if test_size is null the test_train_ratio will be used
data_source_dir = "Track1-RGB/Track1-RGB256x256"

random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

keras.utils.plot_model(model, to_file="test_keras_plot_model.png", show_shapes=True)
IPython.display.Image("test_keras_plot_model.png")
