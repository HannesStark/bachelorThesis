import tifffile
import numpy as np
import tensorflow as tf
rgbPic="C:\\Users\\Hannes\\projects\\bachelorarbeit\\Test-Track1\\JAX_160_001_RGB.tif"
msiPic="C:\\Users\\Hannes\\projects\\ba%pchelorarbeit\\Test-Track1\\JAX_160_001_MSI.tif"

init = tf.global_variables_initializer()
tiffimg = tifffile.imread(rgbPic)
#plt.imshow(tiffimg)
#plt.show()

x= tf.Variable(4, name="x")
y = tf.Variable(6, name="y")
f = x*y*4

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    print(f.eval())
np.c_