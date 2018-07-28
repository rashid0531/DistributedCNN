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
import Model.MCNN as model
import os
import re
import time
from datetime import datetime


# Source:
# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def training_dataset(args):

    train_set_image, train_set_gt, test_set_image, test_set_gt = prepare.get_train_test_DataSet(args["image_path"], args["gt_path"], args["dataset_train_test_ratio"])

    print(len(train_set_image) , len(train_set_gt))

    # A vector of filenames for trainset.
    images_input_train = tf.constant(train_set_image)
    images_gt_train = tf.constant(train_set_gt)

    dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_gt_train))
    # At time of this writing Tensorflow doesn't support a mixture of user defined python function with tensorflow operations.
    # So we can't use one py_func to process data using tenosrflow operation and nontensorflow operation.

    Batched_dataset_train = dataset_train.map(
        lambda img, gt: tf.py_func(prepare.read_npy_file, [img, gt], [img.dtype, tf.float32]))

    Batched_dataset_train = Batched_dataset_train \
        .shuffle(buffer_size=4000) \
        .map(prepare._parse_function,num_parallel_calls= args["num_parallel_threads"]) \
        .batch(batch_size = args["batch_size"])\
        .prefetch(buffer_size = args["prefetch_buffer"])\
        .repeat()

    return Batched_dataset_train


def core_model(input_image):
    mcnn_model = model.MCNN(input_image)
    predicted_density_map = mcnn_model.final_layer_output
    return predicted_density_map


def training_model(input_fn):
    inputs = input_fn()
    image = inputs[0]
    gt = inputs[1]
    predicted_density_map = core_model(image)

    # I am planning to train the backpropagation based on the loss between two 4D arrays. But while showcasing the result, the predictibility will be measure based on mse.
    # Because from my understanding this is what the author did in the paper MCNN.
    cost = tf.reduce_mean(
        (tf.reduce_sum(tf.square(tf.subtract(gt, predicted_density_map)), axis=[1, 2, 3], keepdims=True)))
    # cost = tf.losses.mean_squared_error(gt, predicted_density_map)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)

    sum_of_gt = tf.reduce_sum(gt, axis=[1, 2, 3], keepdims=True)
    sum_of_predicted_density_map = tf.reduce_sum(predicted_density_map, axis=[1, 2, 3], keepdims=True)

    mse = tf.sqrt(tf.reduce_mean(tf.square(sum_of_gt - sum_of_predicted_density_map)))

    mae = tf.reduce_mean(
        tf.reduce_sum(tf.subtract(sum_of_gt, sum_of_predicted_density_map), axis=[1, 2, 3], keepdims=True))

    # Adding summary to the graph.
    # added a small threshold value with mae to prevent NaN to be stored in summary histogram.
    tf.summary.scalar("Mean Squared Error", mae + 1e-8)
    # tf.summary.scalar("loss", cost)

    return cost, mae


def do_training(args, update_op, loss, summary):
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # tf log initialization.
        currenttime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        logdir = "{}/run-{}/".format(args["log_path"], currenttime)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        # !!!!!!
        step = 0
        for step in range(0, args["max_steps"]):

            start_time = time.time()
            _, loss_value = sess.run((update_op, loss))
            duration = time.time() - start_time
            if step % 10 == 0:
                # num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus

                num_examples_per_step = args["batch_size"] * args["num_gpus"]
                examples_per_sec = num_examples_per_step / duration

                sec_per_batch = duration / args["num_gpus"]

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                summary_str = sess.run(summary,
                                       options=run_options,
                                       run_metadata=run_metadata)

                summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
                summary_writer.add_summary(summary_str, step)
                # print('Adding run metadata for', step)

                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open('timeline_02_step_%d.json' % step, 'w') as f:
                #     f.write(chrome_trace)

    print('Final loss: {}'.format(loss_value))


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.
    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.
    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


# Source:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def create_parallel_optimization(args,model_fn, input_fn, optimizer, controller="/cpu:0"):
#def create_parallel_optimization(args,model_fn, input_fn, optimizer, controller="/GPU:0"):

    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`

    devices = get_available_gpus()

    devices = devices[:args["num_gpus"]]

    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []
    mean_squared_err = []

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name) as scope:
                # Compute loss and gradients, but don't apply them yet
                loss, mse = model_fn(input_fn)

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

                losses.append(loss)
                mean_squared_err.append(mse)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()

        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)


        # !!!!!!!

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    avg_loss = tf.reduce_mean(losses)
    avg_mse = tf.reduce_mean(mean_squared_err)

    # !!!!!!!
    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    return apply_gradient_op, avg_mse, summary_op


def parallel_training(args,model_fn, dataset):
    iterator = dataset.make_one_shot_iterator()

    def input_fn():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()

    optimizer = tf.train.AdamOptimizer(learning_rate=args["learning_rate"])
    update_op, loss, summary = create_parallel_optimization(args,
                                                            model_fn,
                                                            input_fn,
                                                            optimizer)

    do_training(args,update_op, loss, summary)


if __name__ == "__main__":

    # The following default values will be used if not provided from the command line arguments.
    DEFAULT_NUMBER_OF_GPUS = 4
    DEFAULT_MAXSTEPS = 21000
    DEFAULT_BATCHSIZE = 64
    DEFAULT_PARALLEL_THREADS = 8
    DEFAULT_PREFETCH_BUFFER_SIZE = DEFAULT_BATCHSIZE * DEFAULT_NUMBER_OF_GPUS * 1
    DEFAULT_IMAGE_PATH = "/home/mrc689/Sampled_Dataset"
    DEFAULT_GT_PATH = "/home/mrc689/Sampled_Dataset_GT/density_map"
    DEFAULT_LOG_PATH = "/home/mrc689/tf_logs"
    DEFAULT_RATIO_TRAINTEST_DATASET = 0.7
    DEFAULT_LEARNING_RATE = 0.001


    # Create arguements to parse
    ap = argparse.ArgumentParser(description="Script to train the FlowerCounter model using multiGPUs in single node.")

    ap.add_argument("-g", "--num_gpus", required=False, help="How many GPUs to use.",default = DEFAULT_NUMBER_OF_GPUS) 
    ap.add_argument("-b", "--batch_size", required=False, help="Number of images to process in a batch per GPU",default = DEFAULT_BATCHSIZE)
    ap.add_argument("-steps", "--max_steps", required=False, help="Maximum number of batches to run.", default = DEFAULT_MAXSTEPS)
    ap.add_argument("-i", "--image_path", required=False, help="Input path of the images",default = DEFAULT_IMAGE_PATH)
    ap.add_argument("-gt", "--gt_path", required=False, help="Ground truth path of input images",default = DEFAULT_GT_PATH)
    ap.add_argument("-num_threads", "--num_parallel_threads", required=False, help="Number of threads to use in parallel for preprocessing elements in input pipeline", default = DEFAULT_PARALLEL_THREADS)
    ap.add_argument("-l", "--log_path", required=False, help="Path to save the tensorflow log files",default=DEFAULT_LOG_PATH)
    ap.add_argument("-r", "--dataset_train_test_ratio", required=False, help="Dataset ratio for train and test set .",default = DEFAULT_RATIO_TRAINTEST_DATASET) 
    ap.add_argument("-pbuff","--prefetch_buffer",required=False,help="An internal buffer to prefetch elements from the input dataset ahead of the time they are requested",default=DEFAULT_PREFETCH_BUFFER_SIZE)
    ap.add_argument("-lr", "--learning_rate", required=False, help="Default learning rate.",default = DEFAULT_LEARNING_RATE)

    args = vars(ap.parse_args())


    start_time = time.time()
    tf.reset_default_graph()

    parallel_training(args,training_model, training_dataset(args))
    duration = time.time() - start_time

    print("Duration : ", duration)

