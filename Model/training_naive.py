#!/usr/bin/env python
import argparse
import tensorflow as tf
import os
from PIL import Image, ImageFile
import numpy as np
import prepare
from matplotlib import pyplot as plt
from datetime import datetime
from tensorflow.python.client import timeline
import modified_MCNN as model
import os
import re
import time
from datetime import datetime
import subprocess
import psutil
from tensorflow.contrib import slim
import analyze_ops_and_vars as analyzer

tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def core_model(input_image):
    mcnn_model = model.MCNN(input_image)
    predicted_density_map = mcnn_model.final_layer_output
    return predicted_density_map


def do_training(args):

    
    train_set_image, train_set_gt, test_set_image, test_set_gt = prepare.get_train_test_DataSet(args["image_path"], args["gt_path"], args["dataset_train_test_ratio"])
    
    TRAINSET_LENGTH = len(train_set_image)

    # print("Trainset Length : ", len(train_set_image) , len(train_set_gt))

    images_input_train = tf.constant(train_set_image)
    images_gt_train = tf.constant(train_set_gt)

    dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_gt_train))
    # At time of this writing Tensorflow doesn't support a mixture of user defined python function with tensorflow operations.
    # So we can't use one py_func to process data using tenosrflow operation and nontensorflow operation.

    # Train Set
    Batched_dataset_train = dataset_train.map(
        lambda img, gt: tf.py_func(prepare.read_npy_file, [img, gt], [img.dtype, tf.float32]))


    Batched_dataset_train = Batched_dataset_train \
        .shuffle(buffer_size=500) \
        .map(prepare._parse_function,num_parallel_calls= args["num_parallel_threads"]) \
        .apply(tf.contrib.data.batch_and_drop_remainder(args["batch_size_per_GPU"])) \
        .prefetch(buffer_size = args["prefetch_buffer"])\
        .repeat()


    iterator = Batched_dataset_train.make_one_shot_iterator()

    mini_batch = iterator.get_next()

    image_names = mini_batch[0]

    # If the number of GPUs is set to 1, then no splitting will be done
    split_batches_imgs = tf.split(mini_batch[1], int(args["num_gpus"]))
    split_batches_gt = tf.split(mini_batch[2], int(args["num_gpus"]))

    predicted_density_map = core_model(split_batches_imgs[0])

    cost = tf.reduce_mean(
            tf.sqrt(
            tf.reduce_sum(
            tf.square(
            tf.subtract(split_batches_gt[0], predicted_density_map)), axis=[1, 2, 3], keepdims=True)))


    sum_of_gt = tf.reduce_sum(split_batches_gt[0], axis=[1, 2, 3], keepdims=True)
    sum_of_predicted_density_map = tf.reduce_sum(predicted_density_map, axis=[1, 2, 3], keepdims=True)

    mse = tf.sqrt(tf.reduce_mean(tf.square(sum_of_gt - sum_of_predicted_density_map)))

    # Changed the mean abosolute error.
    mae = tf.reduce_mean(
        tf.reduce_sum(tf.abs(tf.subtract(sum_of_gt, sum_of_predicted_density_map)), axis=[1, 2, 3], keepdims=True),name="mae")

    # Adding summary to the graph.
    # added a small threshold value with mae to prevent NaN to be stored in summary histogram.
    #tf.summary.scalar("Mean Squared Error", mse)
    tf.summary.scalar("Mean_Absolute_Error", mae)

    # Retain the summaries from the final tower.
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    
    analyzer.analyze_vars(tf.trainable_variables(), print_info=True)
    
    #slim.model_analyzer.analyze_vars([my_var], print_info=True)

    summary_op = tf.summary.merge(summaries)

    global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate=(args["learning_rate"]))

    train_op = optimizer.minimize(cost, global_step=global_step)


    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    arr_examples_per_sec = []

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        # tf log initialization.
        currenttime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        logdir = "{}/run-{}/".format(args["log_path"], currenttime)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        end_point = int((TRAINSET_LENGTH * int(args["number_of_epoch"])) / int(args["batch_size"]))

        for step in range(0, end_point):

            start_time = time.time()
            _, loss_value = sess.run((train_op, mae))

            duration = time.time() - start_time
            if step % 10 == 0:
                # num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus

                num_examples_per_step = args["batch_size"]
                examples_per_sec = num_examples_per_step / duration

                arr_examples_per_sec.append(examples_per_sec)

                sec_per_batch = duration / args["num_gpus"]

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

    print('Final loss: {}'.format(loss_value))

    

if __name__ == "__main__":

    # The following default values will be used if not provided from the command line arguments.
    DEFAULT_NUMBER_OF_GPUS = 1
    DEFAULT_EPOCH = 5999
    
    DEFAULT_NUMBER_OF_WORKERS = 3
    DEFAULT_NUMBER_OF_PS = 1
    DEFAULT_BATCHSIZE_PER_GPU = 16

    DEFAULT_BATCHSIZE = DEFAULT_BATCHSIZE_PER_GPU * DEFAULT_NUMBER_OF_GPUS * DEFAULT_NUMBER_OF_WORKERS
    DEFAULT_PARALLEL_THREADS = 8
    DEFAULT_PREFETCH_BUFFER_SIZE = DEFAULT_BATCHSIZE * DEFAULT_NUMBER_OF_GPUS * 2
    DEFAULT_IMAGE_PATH = "/home/mrc689/Sampled_Dataset"
    DEFAULT_GT_PATH = "/home/mrc689/Sampled_Dataset_GT/density_map"
    DEFAULT_LOG_PATH = "/home/mrc689/tf_logs"
    DEFAULT_RATIO_TRAINTEST_DATASET = 0.7
    DEFAULT_LEARNING_RATE = 0.00001
    DEFAULT_CHECKPOINT_PATH = "/home/mrc689/tf_ckpt"
    DEFAULT_LOG_FREQUENCY = 10

    #DEFAULT_MAXSTEPS = (DEFAULT_TRAINSET_LENGTH * DEFAULT_EPOCH) / DEFAULT_BATCHSIZE

    # Create arguements to parse
    ap = argparse.ArgumentParser(description="Script to train the FlowerCounter model using multiGPUs in single node.")

    ap.add_argument("-g", "--num_gpus", required=False, help="How many GPUs to use.",default = DEFAULT_NUMBER_OF_GPUS)
    ap.add_argument("-e", "--number_of_epoch", required=False, help="Number of epochs",default = DEFAULT_EPOCH)
    ap.add_argument("-b", "--batch_size", required=False, help="Number of images to process in a minibatch",default = DEFAULT_BATCHSIZE)
    ap.add_argument("-gb", "--batch_size_per_GPU", required=False, help="Number of images to process in a batch per GPU",default = DEFAULT_BATCHSIZE_PER_GPU)
    ap.add_argument("-i", "--image_path", required=False, help="Input path of the images",default = DEFAULT_IMAGE_PATH)
    ap.add_argument("-gt", "--gt_path", required=False, help="Ground truth path of input images",default = DEFAULT_GT_PATH)
    ap.add_argument("-num_threads", "--num_parallel_threads", required=False, help="Number of threads to use in parallel for preprocessing elements in input pipeline", default = DEFAULT_PARALLEL_THREADS)
    ap.add_argument("-l", "--log_path", required=False, help="Path to save the tensorflow log files",default=DEFAULT_LOG_PATH)
    ap.add_argument("-r", "--dataset_train_test_ratio", required=False, help="Dataset ratio for train and test set .",default = DEFAULT_RATIO_TRAINTEST_DATASET) 
    ap.add_argument("-pbuff","--prefetch_buffer",required=False,help="An internal buffer to prefetch elements from the input dataset ahead of the time they are requested",default=DEFAULT_PREFETCH_BUFFER_SIZE)
    ap.add_argument("-lr", "--learning_rate", required=False, help="Default learning rate.",default = DEFAULT_LEARNING_RATE)
    ap.add_argument("-ckpt_path", "--checkpoint_path", required=False, help="Path to save the Tensorflow model as checkpoint file.",default = DEFAULT_CHECKPOINT_PATH)
    

    args = vars(ap.parse_args())


    start_time = time.time()
    tf.reset_default_graph()

    do_training(args)
    duration = time.time() - start_time

    print("Duration : ", duration)
