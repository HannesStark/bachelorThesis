import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile as tiff


t = np.arange(0.0, 2.0, 0.01)

s1 = np.sin(2 * np.pi * t)
s2 = np.exp(-t)
s3 = s1 * s2


pred_on = list()
file_names = os.listdir("RGB-From-Track1128x128split8")
for filename in file_names[0:5]:
    pred_on.append(tiff.imread("RGB-From-Track1128x128split8/" + filename))
pred_on = np.array(pred_on, dtype=np.float32)
pred_on /= 255.

fig, axs = plt.subplots(3, 10, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

for i in range(0,3):
    for j in range(0,10):
        axs[i,j].imshow(pred_on[1])
        axs[i,j].axis('off')


plt.show()