from pathlib import Path
from preprocessing.load_mock_dataset import load_training_image_groundtruth
import tensorflow as tf
from model import config
import numpy as np
import cv2

DEFAULT_IMAGE_PATH = Path('./mock_data/training_images/1109-0704/')
DEFAULT_GT_PATH = Path('./mock_data/ground_truth/1109-0704/')
input_shape = (224, 224, 3)
DEFAULT_BATCHSIZE_PER_GPU = 8
DEFAULT_PREFETCH_BUFFER_SIZE = DEFAULT_BATCHSIZE_PER_GPU * 4


def load_ny_file_as_eager_tensor(gt_ny_file_path):
    gt_data = np.load(gt_ny_file_path.decode())
    gt_data = cv2.resize(gt_data, (config.input_image_width, config.input_image_width))
    gt_data = np.reshape(gt_data, [gt_data.shape[1], gt_data.shape[0], 1])
    gt_data = gt_data.astype(np.float32)
    return gt_data


def read_img_numpy_files(img_path, gt_path):
    image_string = tf.io.read_file(img_path)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize(image, [config.input_image_width, config.input_image_width])

    # https://stackoverflow.com/questions/61891990/how-to-load-npy-files-from-different-directories-in-tensorflow-data-pipeline-fr
    return resized_image, tf.py_function(load_ny_file_as_eager_tensor, inp=[gt_path], Tout=[tf.float32])


train_images, train_gts = load_training_image_groundtruth(DEFAULT_IMAGE_PATH, DEFAULT_GT_PATH)
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_gts))
dataset = dataset.shuffle(len(train_images)). \
    map(read_img_numpy_files, num_parallel_calls=4). \
    batch(8, drop_remainder=True). \
    prefetch(128). \
    repeat()

# Continue from chapter 7.3
