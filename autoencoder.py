import tifffile
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime

rgbPic = "C:\\Users\\Hannes\\projects\\bachelorarbeit\\Test-Track1\\JAX_160_001_RGB.tif"
msiPic = "C:\\Users\\Hannes\\projects\\ba%pchelorarbeit\\Test-Track1\\JAX_160_001_MSI.tif"

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

tiffimg = tifffile.imread(rgbPic)
# plt.imshow(tiffimg)
# plt.show()
hist = cv2.calcHist([tiffimg], [0],None, [1024], [0,256])

nlearning_rate = 0.01

x = tf.Variable(4, name="x")
y = tf.Variable(6, name="y")
f = x * y * 4
z = x + tf.Variable(f)
x_summary = tf.summary.scalar("name",y)
filewriter = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        result1, result2 = sess.run([f,z])

print(result1)
print(result2)
