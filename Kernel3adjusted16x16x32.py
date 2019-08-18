import tensorflow as tf
from keras import losses
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np
import os
import tifffile as tiff
from matplotlib import pyplot as plt
import time

K = tf.keras.backend

latent_dim = 1024
file_tag = "16x16x32TRAINEDON178000"
# file_tag = "16x16x32TRAINEDON178000" + "_dim" + str(latent_dim)
# file_tag = os.path.splitext(os.path.basename(__file__))[0] + "_dim" + str(latent_dim)
# file_tag = "test_with_unscaled_latent_loss16x16x32"
batch_size = 128
epochs = 50
num_examples_to_generate = 16
test_train_ratio = 1 / 8  # is only used if test_size is null
test_size = 200  # if test_size is null the test_train_ratio will be used
generation_path = "generatedImages/" + file_tag + "/"

# data_source_dir = "Track1-RGB/Track1-RGB128x128"
data_source_dir = "test-RGB128x128split8"
truth_source_dir = "Track1-Truth"


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


inputs = tf.keras.layers.Input(shape=[128, 128, 3])
z = tf.keras.layers.Conv2D(8, strides=2, kernel_size=3, padding="same", activation="selu")(inputs)
z = tf.keras.layers.Conv2D(16, strides=2, kernel_size=3, padding="same", activation="selu")(z)
z = tf.keras.layers.Conv2D(32, strides=2, kernel_size=3, padding="same", activation="selu")(z)

z = tf.keras.layers.Flatten()(z)
codings_mean = tf.keras.layers.Dense(latent_dim)(z)  # μ
codings_log_var = tf.keras.layers.Dense(latent_dim)(z)  # γ
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

decoder_inputs = tf.keras.layers.Input(shape=[latent_dim])
x = tf.keras.layers.Dense(16 * 16 * 32)(decoder_inputs)
x = tf.keras.layers.Reshape([16, 16, 32])(x)
x = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="selu")(x)
x = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding="same", activation="selu")(x)
outputs = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid")(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)
variational_ae.add_loss(K.mean(latent_loss) / (128. * 128. * 3.))
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

log_file = "log/" + file_tag + "{}.log".format(time.strftime("%d_%m_%Y_%H_%M_%S"))
if not os.path.exists("log"):
    os.mkdir("log")


def train():
    last_epoch_mean = []
    for epoch in range(1, epochs + 1):
        for iteration in range(train_iterations):
            first_index = iteration * batch_size
            batch = get_image_batch(training_paths, first_index, batch_size)
            history = variational_ae.train_on_batch(batch, batch)

            result = str(history) + "\n" + "Epoch: {}/{}...".format(epoch, epochs) + "Iteration: {}/{}...".format(
                iteration + 1, train_iterations) + "Images: {}/{}...".format((iteration + 1) * batch_size,
                                                                             train_iterations * batch_size) + "\n"
            print(result)
            f = open(log_file, "a")
            f.write(result)
            f.close()
            if epoch == epochs:
                last_epoch_mean.append(history)
    last_epoch_mean = np.mean(last_epoch_mean, 0)
    print(last_epoch_mean)
    f = open(log_file, "a")
    f.write(str(last_epoch_mean))
    f.close()


if not os.path.exists(generation_path):
    os.mkdir(generation_path)


def predictions_and_generations():
    pred_on = list()
    file_names = os.listdir("RGB-From-Track1128x128split8")
    for filename in file_names[5:15]:
        pred_on.append(tiff.imread("RGB-From-Track1128x128split8/" + filename))
    pred_on = np.array(pred_on, dtype=np.float32)
    pred_on /= 255.

    latent_log_var, latent_mean, latent_vars = variational_encoder.predict(pred_on)
    predictions = variational_decoder.predict(latent_vars)

    for i in range(0, len(predictions)):
        plt.clf()
        plt.subplot(2, 3, 1)
        plt.title('Latent Log Variance')
        plt.plot(latent_log_var[i], 'o', markersize=1)
        plt.subplot(2, 3, 2)
        plt.title('Latent Mean')
        plt.plot(latent_mean[i], 'o', markersize=1)
        plt.subplot(2, 3, 3)
        plt.title('Latent Sampled Variables')
        plt.plot(latent_vars[i], 'o', markersize=1)
        plt.subplot(2, 3, 4)
        plt.title('Original Image')
        plt.imshow(pred_on[i])
        plt.subplot(2, 3, 5)
        plt.title('Reconstructed Image')
        plt.imshow(predictions[i])

        plt.tight_layout()

        plt.savefig(generation_path + "reconstruction" + str(i))

    codings = tf.random.normal(shape=[12, latent_dim])
    images_generated = variational_decoder.predict(codings, steps=1)
    print(type(images_generated))
    print(images_generated.shape)

    for i in range(0, len(images_generated)):
        plt.clf()
        plt.imshow(images_generated[i])
        plt.savefig(generation_path + "generated" + str(i))


def get_predominant_class_as_color(img):
    flat = img.flatten()
    most_common = np.bincount(flat).argmax()
    if most_common == 2:  # ground
        return 'gray'
    elif most_common == 5:  # vegetation
        return 'g'
    elif most_common == 6:  # building
        return 'r'
    elif most_common == 9:  # water
        return 'b'
    elif most_common == 65:  # clutter
        return 'y'


def tsne_vis():
    predominant_classes = []
    locations = []
    average_heights = []
    images = []
    file_names = os.listdir("RGB-From-Track1128x128split8")
    print(len(file_names))
    for filename in file_names[0:5000]:
        image = tiff.imread("RGB-From-Track1128x128split8/" + filename)
        cls = tiff.imread("Track1-Truth128x128split8/" + filename.replace("RGB", "CLS"))
        predominant_classes.append(get_predominant_class_as_color(cls))
        images.append(image)
        if "OMA" in filename:
            locations.append('.')
        else:
            locations.append('s')

    images = np.array(images, dtype=np.float32)
    images /= 255.

    latent_log_var, latent_mean, latent_vars = variational_encoder.predict(images)
    predictions = variational_decoder.predict(latent_vars)

    print("t-SNE started with: " + str(latent_vars.shape))
    embedded = TSNE(perplexity=15, n_iter=5000).fit_transform(latent_vars)
    # embedded = PCA(n_components=2).fit_transform(latent_vars)
    print("t-SNE finished")

    plt.clf()
    # plt.scatter(embedded[:, 0], embedded[:, 1], s=1, c=predominant_classes)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(0, len(embedded)):
        x = embedded[i, 0]
        y = embedded[i, 1]
        bb = Bbox.from_bounds(x, y, 1, 1)
        bb2 = TransformedBbox(bb, ax.transData)
        bbox_image = BboxImage(bb2,
                               norm=None,
                               origin=None,
                               clip_on=False)
        bbox_image.set_data(images[i])
        ax.add_artist(bbox_image)

    plt.show()


# start_time = time.time()
# train()
# f = open(log_file, "a")
# f.write(str(time.time() - start_time))
# f.close()
# variational_ae.save_weights("./savedModels/" + file_tag + ".h5")

variational_ae.load_weights("./savedModels/" + file_tag + ".h5")

# predictions_and_generations()

tsne_vis()
