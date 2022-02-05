from matplotlib import pyplot as plt
import numpy as np

which_numpu_arrays = ["/home/mrc689/Sampled_Dataset_GT/density_map/xavier/1109-0802/frame000231_1_3.npy",
"/home/mrc689/Sampled_Dataset_GT/density_map/xavier/1109-0802/frame000418_1_2.npy","/home/mrc689/Sampled_Dataset_GT/density_map/xavier/1109-0802/frame000154_1_4.npy"]

arr = np.load(which_numpu_arrays[0])
plt.imsave('man21.png',arr)
print(np.sum(arr))
#plt.show()

arr = np.load(which_numpu_arrays[1])
plt.imsave('man5.png', arr)
print(np.sum(arr))
#plt.show()

arr = np.load(which_numpu_arrays[2])
print(np.sum(arr))
plt.imsave('manx.png', arr)

