from PIL import Image, ImageFile
import numpy as np
import cv2
import tensorflow as tf

import config as config
import multicolumn_cnn as mcnn

def read_npy_file(item):

    # The ground truth density map needs to be downsampled because after beign processed through the MAX-POOL layers the input is downsized in half for each MAX-POOL layer.
    data = np.load(item)
    width =  int(config.input_image_width/4)
    height = int(config.input_image_height/4)
    data = cv2.resize(data, (width, height))
    data = data * ((width * height) / (width * height))

    # !!!!!!!!!!!!!!!! This reshaping doesn't need to be done if the density map is multichanneled. !!!!!!!!!!!!!!!!!!!!!!
    # data = np.reshape(data, [data.shape[1], data.shape[0], 1])
    return data.astype(np.float32)

def read_image_using_PIL(image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(image)
    image = image.resize((224,224))
    image = np.asarray(image, np.uint8)
    # print(np.sum(image))
    # img = Image.fromarray(image)
    # img.show()
    return image

if __name__ == "__main__":
    
    X = tf.placeholder(tf.float32, [1, 224, 224, 3])
    ob1 = mcnn(X)
    
    input_path = "/media/mohammed/Drive_full_of_surprises/Projects/Dataset/Image_Tiles/1237/part2/1237-0725/frame000001_0_0.jpg"

    image = read_image_using_PIL(input_path)

    with tf.Session() as sess:
    # Initialize all variables
        sess.run(tf.global_variables_initializer())
        output = sess.run(ob1.final_layer_output,feed_dict={X: [image]})
    # # #
    print(np.shape(output))