import numpy as np
from matplotlib import pyplot as plt
import multicolumn_cnn
from pathlib import Path

from util import util

if __name__ == "__main__":

    input_shape = (224, 224, 3)
    test_data_folder = Path('./Model/test_images')
    test_img = test_data_folder / 'frame001284_1_0.jpg'
    image = util.read_image_using_PIL(test_img)
    MCNN = multicolumn_cnn.get_model(input_shape)
    prediction = MCNN(np.asarray([image]))
    plt.imshow(prediction[0])
    plt.show()