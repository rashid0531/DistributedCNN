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

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

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
    
    print(len(test_set_image) , len(test_set_gt))

    # A vector of filenames for testset
    images_input_test = tf.constant(test_set_image) 
    images_gt_test = tf.constant(test_set_gt)
    
    # At time of this writing Tensorflow doesn't support a mixture of user defined python function with tensorflow operations.
    # So we can't use one py_func to process data using tenosrflow operation and nontensorflow operation.

    dataset_test = tf.data.Dataset.from_tensor_slices((images_input_test, images_gt_test))
    Batched_dataset_test = dataset_test.map(
        lambda img, gt: tf.py_func(prepare.read_npy_file, [img, gt], [img.dtype, tf.float32]))

    Batched_dataset_test = Batched_dataset_test \
        .map(prepare._parse_function,num_parallel_calls= args["num_parallel_threads"]) \
        .batch(batch_size = args["batch_size"])\
        .prefetch(buffer_size = args["prefetch_buffer"])\
        .repeat()

    return Batched_dataset_test


def core_model(input_image):
    mcnn_model = model.MCNN(input_image)
    predicted_density_map = mcnn_model.final_layer_output
    return predicted_density_map


def training_model(input_img, ground_truth):

    image = input_img
    gt = ground_truth
    predicted_density_map = core_model(image)

    # Evaluation of the model is calculated using relative error.
    sum_of_gt = tf.reduce_sum(gt, axis=[1, 2, 3], keepdims=True)
    sum_of_predicted_density_map = tf.reduce_sum(predicted_density_map, axis=[1, 2, 3], keepdims=True)

    relative_error = tf.divide(tf.abs(tf.subtract(sum_of_predicted_density_map , sum_of_gt)), sum_of_gt)

    return relative_error, sum_of_gt, sum_of_predicted_density_map


def do_training(args,loss,image_names,ground_truth, predictions):
    
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    #proc = subprocess.Popen(['./scr'])
    #print("start process with pid %s" % proc.pid)

    with tf.Session(config=config) as sess:
        
        saver.restore(sess, "/home/mohammed/tf_ckpt/checkpoint@step-19000.ckpt")

        # tf log initialization.
        currenttime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        logdir = "{}/run-{}/".format(args["log_path"], currenttime)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        logs = []

        step = 0
        for step in range(0, args["max_steps"]):

            start_time = time.time()
            loss_value, img_names, original_count, predicted_count = sess.run([loss,image_names, ground_truth, predictions])
            loss_value = np.array(loss_value)
            img_names = np.array(img_names)
            original_count = np.array(original_count)
            predicted_count = np.array(predicted_count)
            
            #print(original_count.shape, predicted_count.shape)

            shape = img_names.shape
           
            for gpu_id in range(0,shape[0]):
                
                for img in range(0,shape[1]):
                    
                    img_names[gpu_id][img] = str(img_names[gpu_id][img])
                    img_prefix = img_names[gpu_id][img].split('/')[-2:]
                    img_prefix = '/'.join(img_prefix)

                    output = "IMG: {}, Original Count: {:07.4f}, Predicted Count: {:07.4f}, Relative Error: {:07.4f}" \
                             .format(img_prefix, original_count[gpu_id][img], predicted_count[gpu_id][img], loss_value[gpu_id][img]) 
            
                    logs.append(output)

                    """
                    print("IMG: ", img_prefix,
                          "Original Count: %.3f ," %original_count[gpu_id][img], 
                          "Predicted Count: %.3f ," % predicted_count[gpu_id][img],
                          "Relative Error: %.3f" %loss_value[gpu_id][img])
                    """
                    print(output)

        duration = time.time() - start_time

        with open("output.txt","w+") as file_object:
            for i in range(0,len(logs)):
                file_object.write(logs[i]+"\n")
         
    #kill(proc.pid)


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


def evaluate_test_set(args,model_fn,input_fn,controller="/cpu:0"):
   
    devices = get_available_gpus()

    devices = devices[:args["num_gpus"]]

    relative_err = []
    ground_truth = []
    predictions = []

    # Get the next mini batch from the iterator.

    mini_batch = input_fn()

    #image_names = mini_batch[0]

    #print(type(output[0]))
    split_batches_img_names = tf.split(mini_batch[0], int(args["num_gpus"])) 
    split_batches_imgs = tf.split(mini_batch[1], int(args["num_gpus"]))
    split_batches_gt = tf.split(mini_batch[2], int(args["num_gpus"]))

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name) as scope:
                # Compute relative error
                re , gt , prediction = model_fn(split_batches_imgs[i],split_batches_gt[i])

                # Gradient computation should be turned off for testing dataset as I dont want to train the model from the testing dataset. 
                """
                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                """
                relative_err.append(re)
                ground_truth.append(gt)
                predictions.append(prediction)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

    # relative_err is 5 dimentional. 
    # The first index indicates the GPU ID from which the result was generated, the second one is the batch id, the third ,fourth and fifth  are the height,width and channels respectively.           

    relative_err = tf.reshape(relative_err,[args["num_gpus"], -1])
    ground_truth = tf.reshape(ground_truth,[args["num_gpus"], -1])
    predictions = tf.reshape(predictions,[args["num_gpus"], -1])

    return relative_err,split_batches_img_names, ground_truth, predictions


def parallel_training(args,model_fn, dataset):
    iterator = dataset.make_one_shot_iterator()

    def input_fn():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()

    # No optimizer is needed for testing phase.
    #optimizer = tf.train.AdamOptimizer(learning_rate=args["learning_rate"])
    
    loss, image_names, ground_truth, predictions = evaluate_test_set(args,model_fn,input_fn)

    do_training(args,loss,image_names, ground_truth, predictions)


if __name__ == "__main__":

    # The following default values will be used if not provided from the command line arguments.
    DEFAULT_NUMBER_OF_GPUS = 1
    DEFAULT_MAXSTEPS = 114
    DEFAULT_BATCHSIZE_PER_GPU = 32
    DEFAULT_BATCHSIZE = DEFAULT_BATCHSIZE_PER_GPU * DEFAULT_NUMBER_OF_GPUS
    DEFAULT_PARALLEL_THREADS = 8
    DEFAULT_PREFETCH_BUFFER_SIZE = DEFAULT_BATCHSIZE * DEFAULT_NUMBER_OF_GPUS * 1
    DEFAULT_IMAGE_PATH = "/home/mrc689/Sampled_Dataset"
    DEFAULT_GT_PATH = "/home/mrc689/Sampled_Dataset_GT/density_map"
    DEFAULT_LOG_PATH = "/home/mrc689/tf_logs"
    DEFAULT_RATIO_TRAINTEST_DATASET = 0.7
    DEFAULT_LEARNING_RATE = 0.00001
    DEFAULT_CHECKPOINT_PATH = "/home/mrc689/tf_ckpt"

    # Create arguements to parse
    ap = argparse.ArgumentParser(description="Script to train the FlowerCounter model using multiGPUs in single node.")

    ap.add_argument("-g", "--num_gpus", required=False, help="How many GPUs to use.",default = DEFAULT_NUMBER_OF_GPUS)
    ap.add_argument("-b", "--batch_size", required=False, help="Number of images to process in a minibatch",default = DEFAULT_BATCHSIZE)
    ap.add_argument("-gb", "--batch_size_per_GPU", required=False, help="Number of images to process in a batch per GPU",default = DEFAULT_BATCHSIZE_PER_GPU)
    ap.add_argument("-steps", "--max_steps", required=False, help="Maximum number of batches to run.", default = DEFAULT_MAXSTEPS)
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



    proc = subprocess.Popen(['./scr'])
    print("start process with pid %s" % proc.pid)

    parallel_training(args,training_model, training_dataset(args))
    duration = time.time() - start_time

    kill(proc.pid)

    print("Duration : ", duration)

