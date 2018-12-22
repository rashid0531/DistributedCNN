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

tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# This function kills all the child processes associated with the parent process sent as function argument.
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def core_model(input_image):
    mcnn_model = model.MCNN(input_image)
    predicted_density_map = mcnn_model.final_layer_output
    return predicted_density_map


def do_training(args):

    ps_hosts = args["ps_hosts"].split(",")
    worker_hosts = args["worker_hosts"].split(",")

    print("Number of Parameter Servers: ",len(ps_hosts), " Number of workers: ", len(worker_hosts))

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=args["job_name"],
                           task_index=int(args["task_index"]))

    if args["job_name"] == "ps":
        server.join()

    elif args["job_name"] == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % int(args["task_index"]), ps_device="/job:ps/cpu:0", cluster=cluster)):

            train_set_image, train_set_gt, test_set_image, test_set_gt = prepare.get_train_test_DataSet(args["image_path"], args["gt_path"], args["dataset_train_test_ratio"])

            TRAINSET_LENGTH = len(train_set_image)

            print("Trainset Length : ", len(train_set_image) , len(train_set_gt))

            images_input_train = tf.constant(train_set_image)
            images_gt_train = tf.constant(train_set_gt)

            dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_gt_train))
            # At time of this writing Tensorflow doesn't support a mixture of user defined python function with tensorflow operations.
            # So we can't use one py_func to process data using tenosrflow operation and nontensorflow operation.

            # Train Set
            Batched_dataset_train = dataset_train.map(
                lambda img, gt: tf.py_func(prepare.read_npy_file, [img, gt], [img.dtype, tf.float32]))

            #Batched_dataset_train = Batched_dataset_train.shard(len(worker_hosts),int(args["task_index"]))

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

            summary_op = tf.summary.merge(summaries)

            global_step = tf.train.get_or_create_global_step()

            optimizer = tf.train.AdamOptimizer(learning_rate=(args["learning_rate"]))

            #train_op = optimizer.minimize(cost, global_step=global_step)

            # Synchronous training.
            opt = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=len(worker_hosts),
                               total_num_replicas=len(worker_hosts))

            # Some models have startup_delays to help stabilize the model but when using
            # sync_replicas training, set it to 0.

            # Now you can call `minimize()` or `compute_gradients()` and
            # `apply_gradients()` normally

            # !!!! Doubtful
            training_op = opt.minimize(cost, global_step=global_step)

            #config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config = tf.ConfigProto()
            config.log_device_placement=False
            config.allow_soft_placement=True
            #config.intra_op_parallelism_threads = 44
            #config.inter_op_parallelism_threads = 44

        # The StopAtStepHook handles stopping after running given steps.
        #SHARDED_TRAINSET_LENGTH = int(TRAINSET_LENGTH/len(worker_hosts))
        effective_batch_size = int(args["batch_size_per_GPU"])*len(worker_hosts)  

        end_point = int((TRAINSET_LENGTH * int(args["number_of_epoch"])) / effective_batch_size)
        print("End Point : ",end_point)

        is_chief=(int(args["task_index"]) == 0)

        # You can create the hook which handles initialization and queues.
        sync_replicas_hook = opt.make_session_run_hook(is_chief)
        # Creating profiler hook. 
        profile_hook = tf.train.ProfilerHook(save_steps=1500, output_dir='/home/rashid/DistributedCNN/Model/timeline/')
        # Simple Example of logging hooks

        """
        tensors_to_log = {"MAE ": "mae"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)
        """

        # last_step should be equal to the end_point
        hooks=[sync_replicas_hook, tf.train.StopAtStepHook(last_step=end_point),profile_hook]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.

        arr_examples_per_sec = []

        with tf.train.MonitoredTrainingSession(master=server.target,
                                            is_chief=(int(args["task_index"]) == 0),
                                            checkpoint_dir=args["checkpoint_path"],
                                            hooks=hooks, config=config) as mon_sess:

            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                start_time = time.time()
                _, loss_value, step = mon_sess.run((training_op, mae,global_step))
                duration = time.time() - start_time

                examples_per_sec = (args["batch_size_per_GPU"] * len(worker_hosts)) / duration

                arr_examples_per_sec.append(examples_per_sec)

                format_str = ('%s: step %d, loss = %.2f , examples/sec =  %.1f ')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec))
                
                #mon_sess.run(training_op)


    print("--- Experiment Finished ---") 

    

if __name__ == "__main__":

    # The following default values will be used if not provided from the command line arguments.
    DEFAULT_NUMBER_OF_GPUS = 1
    DEFAULT_EPOCH = 3000
    
    DEFAULT_NUMBER_OF_WORKERS = 1
    DEFAULT_NUMBER_OF_PS = 1
    DEFAULT_BATCHSIZE_PER_GPU = 8

    DEFAULT_BATCHSIZE = DEFAULT_BATCHSIZE_PER_GPU * DEFAULT_NUMBER_OF_GPUS * DEFAULT_NUMBER_OF_WORKERS
    DEFAULT_PARALLEL_THREADS = 8
    #DEFAULT_PREFETCH_BUFFER_SIZE = DEFAULT_BATCHSIZE * DEFAULT_NUMBER_OF_GPUS * 2

    DEFAULT_PREFETCH_BUFFER_SIZE = 64
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
    
    # Arguments needed for Distributed Training.
    ap.add_argument("-pshosts", "--ps_hosts", required=False, help="Comma-separated list of hostname:port pairs.")
    ap.add_argument("-wkhosts", "--worker_hosts", required=False, help="Comma-separated list of hostname:port pairs.")
    ap.add_argument("-job", "--job_name", required=False, help="One of 'ps', 'worker'.")
    ap.add_argument("-tsk_index", "--task_index", required=False, help="Index of task within the job.")

    ap.add_argument("-lg_freq", "--log_frequency", required=False, help="Log frequency.",default = DEFAULT_LOG_FREQUENCY)

    args = vars(ap.parse_args())


    start_time = time.time()
    tf.reset_default_graph()

    # This process initiates the GPU profiling script.
    #proc = subprocess.Popen(['./gpu_profile'])
    #print("start GPU profiling process with pid %s" % proc.pid)

    do_training(args)
    duration = time.time() - start_time

    #kill(proc.pid)

    print("Duration : ", duration)
