import modified_MCNN as models
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    path = "/media/mohammed/Drive_full_of_surprises/Projects/Dataset/ground_truth/density_map/part2/1237-0725/frame000001_0_0.npy"
    data=np.load(path)
    print(data.shape)
    plt.imshow(data)
    plt.show()
