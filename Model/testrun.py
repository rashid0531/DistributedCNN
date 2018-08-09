import modified_MCNN as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    
    path = "/u1/rashid/FlowerCounter_Dataset_GroundTruth/density_map/manual/1109-0805/frame000750_1_0.npy" 
    '''
    path = "/media/mohammed/Drive_full_of_surprises/Projects/Dataset/ground_truth/density_map/part2/1237-0725/frame001010_0_2.npy"
    img_path = "/media/mohammed/Drive_full_of_surprises/Projects/Dataset/Image_Tiles/1237/part2/1237-0725/frame001010_0_2.jpg"
    image = Image.open(img_path)
    plt.imshow(image)
    plt.show()
    '''
    data=np.load(path)
    print(data.shape)
    plt.imshow(data)
    plt.show()
