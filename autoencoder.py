import tifffile
import numpy as np
import tensorflow as tf
from datetime import datetime

rgbPic="C:\\Users\\Hannes\\projects\\bachelorarbeit\\Test-Track1\\JAX_160_001_RGB.tif"
msiPic="C:\\Users\\Hannes\\projects\\ba%pchelorarbeit\\Test-Track1\\JAX_160_001_MSI.tif"

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)


init = tf.global_variables_initializer()
tiffimg = tifffile.imread(rgbPic)
#plt.imshow(tiffimg)
#plt.show()

filewriter = tf.summary.FileWriter(logdir, tf.get_default_graph())



x= tf.Variable(4, name="x")
y = tf.Variable(6, name="y")
f = x*y*4


with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    filewriter.add_summary(x)
    print(f.eval())
