#!/usr/bin/env python
import argparse
import tensorflow as tf
import time
from datetime import datetime
import subprocess
import psutil
import os

import Model.prepare as prepare
import Model.modified_MCNN as model

# Sets the threshold for what messages will be logged.
tf.logging.set_verbosity(tf.logging.INFO)

# Sets if any specific GPU needs to be selected.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def kill(proc_pid):
    """
    This function kills all the child processes forked from the parent process sent as function argument.
    :param proc_pid: The process id of the parent process.
    :return:
    """
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def core_model(input_image):
    """
    This function creates a object of the multi-column CNN model which is defined in modified_MCNN module.
    The multi-column CNN is a three column convolutional neural network architecture which takes a batch of images (transformed to tensors)
    as input and produces density maps which show the approximate positions of the canola flowers for the given images.
    :param input_image: [tensors] batch of input images.
    :return: density map: [tensors] predicted density map.
    """
    mcnn_model = model.MCNN(input_image)
    predicted_density_map = mcnn_model.final_layer_output
    return predicted_density_map

def do_training(args,server,cluster_spec):
    """
    Initiates and executes the main training procedures. From loading the images from HDFS or NFS,
    transforming the images into tensors, defining the cost function for backpropagation and feeding
    the input tensors to the core model and finally executing distributed training through synchronous parameter updates
    etc. are the main responsibilities of this function.
    :param args: command line arguments which specify different parameters during training.
    :param cluster_spec: [dict] a dictionary which contains the cluster specification such as job name and ip addresses of the
    machines which will be assigned to the job. The job name is used as key and the list of ip addresses are used as
    values to comprise the dictionary.
    :return:
    """

    # Automatically assigns operations to the suitable device.
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % int(args["task_index"]),
                                                  ps_device="/job:ps/cpu:0",
                                                  cluster=cluster_spec)):

        # Get the list of input images along with corresponding ground-truths which will be used in train, test dataset.
        train_set_image, train_set_gt, _, _ = prepare.get_train_test_DataSet(args["image_path"],
                                                                       args["gt_path"],
                                                                       args["dataset_train_test_ratio"])

        assert len(train_set_image) == len(train_set_gt), "Equal number of ground-truths and input images are required."

        TRAINSET_LENGTH = len(train_set_image)

        print("Trainset Length : ", len(train_set_image), "Ground Truth Length : ", len(train_set_gt))

        images_input_train = tf.constant(train_set_image)
        images_gt_train = tf.constant(train_set_gt)


        dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_gt_train))
        # At time of this writing Tensorflow doesn't support a mixture of user defined python function
        # with tensorflow operations.So we can't use one py_func to process data using tenosrflow operation and
        # nontensorflow operation.

        # Train Set
        Batched_dataset_train = dataset_train.map(
            lambda img, gt: tf.py_func(prepare.read_npy_file, [img, gt], [img.dtype, tf.float32]))

        Batched_dataset_train = Batched_dataset_train \
            .shuffle(buffer_size=500) \
            .map(prepare._parse_function, num_parallel_calls=args["num_parallel_threads"]) \
            .apply(tf.contrib.data.batch_and_drop_remainder(args["batch_size_per_GPU"])) \
            .prefetch(buffer_size=args["prefetch_buffer"]) \
            .repeat()

        # Create a Tensorflow iterator to iterate over the batched dataset.
        iterator = Batched_dataset_train.make_one_shot_iterator()

        # Generate next item from the iterator.
        mini_batch = iterator.get_next()

        # If the number of GPUs is set to 1, then no splitting will be done
        split_batches_imgs = tf.split(mini_batch[1], int(args["num_gpus"]))
        split_batches_gt = tf.split(mini_batch[2], int(args["num_gpus"]))

        predicted_density_map = core_model(split_batches_imgs[0])
        assert tf.shape(predicted_density_map) == [224, 224, 1], "Dimension of the predicted map needs to be 224x224x1"

        # Definition of loss function (Pixel wise euclidean distance between ground-truth and predicted density map).
        # is used here
        cost = tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(
                    tf.square(
                        tf.subtract(split_batches_gt[0], predicted_density_map)), axis=[1, 2, 3], keepdims=True)))

        # sum over the batched data.
        sum_of_gt = tf.reduce_sum(split_batches_gt[0], axis=[1, 2, 3], keepdims=True)
        sum_of_predicted_density_map = tf.reduce_sum(predicted_density_map, axis=[1, 2, 3], keepdims=True)

        # Mean squared error.
        mse = tf.sqrt(tf.reduce_mean(tf.square(sum_of_gt - sum_of_predicted_density_map)))

        # Mean absolute error.
        mae = tf.reduce_mean(
            tf.reduce_sum(
                tf.abs(
                    tf.subtract(sum_of_gt, sum_of_predicted_density_map)),
                axis=[1, 2, 3],
                keepdims=True),
            name="mae")

        # Adding summary to the graph. The summary can be later visualized using Tensorboard.
        tf.summary.scalar("Mean_Absolute_Error", mae)

        # Collect summaries defined during training.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Add all the trainable variables to summary.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Create global steps. Comes useful while tracking the last synchronized step among multiple worker machines
        # during training.
        global_step = tf.train.get_or_create_global_step()

        # Initiate optimizer for updating gradients. Among many optimizers, AdamOptimizer will be used in this project.
        optimizer = tf.train.AdamOptimizer(learning_rate=(args["learning_rate"]))

        # Synchronous distributed optimizer. Only needed for distributed training. Not needed while training in
        # single machine.
        opt = tf.train.SyncReplicasOptimizer(optimizer,
                                             replicas_to_aggregate=len(args["worker_hosts"].split(",")),
                                             total_num_replicas=len(args["worker_hosts"].split(",")))

        training_op = opt.minimize(cost, global_step=global_step)

        # configuration to set if the device placement will be logged or not.
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.allow_soft_placement = True


    effective_batch_size = int(args["batch_size_per_GPU"]) * len(args["worker_hosts"].split(","))

    end_point = int((TRAINSET_LENGTH * int(args["number_of_epoch"])) / effective_batch_size)
    print("End Point : ", end_point)

    # One worker among all the worker machines takes the responsibility of chief worker. The chief worker periodically
    # saves the model parameter in checkpoint file to ensure fault tolerance. Usually, the first worker in the
    # worker list takes the chief role.
    is_chief = (int(args["task_index"]) == 0)

    sync_replicas_hook = opt.make_session_run_hook(is_chief)

    # Creating profiler hook. Used for profiling CPU, GPU usage, runtime of individual operation.
    profile_hook = tf.train.ProfilerHook(save_steps=1500, output_dir='/home/rashid/DistributedCNN/Model/timeline/')

    # The StopAtStepHook handles when to stop training. When the last_step is reached, the training is terminated.
    # last_step should be equal to the end_point.
    hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=end_point), profile_hook]

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
            # mon_sess.run handles AbortedError in case of preempted PS.

            start_time = time.time()
            _, loss_value, step = mon_sess.run((training_op, mae, global_step))
            duration = time.time() - start_time

            examples_per_sec = (args["batch_size_per_GPU"] * len(args["worker_hosts"].split(","))) / duration

            arr_examples_per_sec.append(examples_per_sec)

            format_str = ('%s: step %d, loss = %.2f , examples/sec =  %.1f ')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec))


    print("--- Experiment Finished ---")


def assign_tasks(args):
    """
    This function assigns jobs to the master and worker machines. From the commandline arguments this function
    determines which of the machines will be used as parameter servers and which machines will be used as workers.
    :param args: the command line arguments which specify different parameters during training.
    :return:
    """

    ps_hosts = args["ps_hosts"].split(",")
    worker_hosts = args["worker_hosts"].split(",")

    print("Number of Parameter Servers: ",len(ps_hosts), " Number of workers: ", len(worker_hosts))

    # Create a cluster from the parameter server and worker hosts (Tensorflow specific code).
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=args["job_name"],
                           task_index=int(args["task_index"]))

    if args["job_name"] == "ps":
        server.join()

    elif args["job_name"] == "worker":
        do_training(args,server,cluster)



if __name__ == "__main__":

    # The following default values will be used if command line arguments are not provided.
    DEFAULT_NUMBER_OF_GPUS = 1
    DEFAULT_EPOCH = 3000
    
    DEFAULT_NUMBER_OF_WORKERS = 1
    DEFAULT_NUMBER_OF_PS = 1
    DEFAULT_BATCHSIZE_PER_GPU = 8

    DEFAULT_BATCHSIZE = DEFAULT_BATCHSIZE_PER_GPU * DEFAULT_NUMBER_OF_GPUS * DEFAULT_NUMBER_OF_WORKERS
    DEFAULT_PARALLEL_THREADS = 8

    DEFAULT_PREFETCH_BUFFER_SIZE = 64
    DEFAULT_IMAGE_PATH = "/home/mrc689/Sampled_Dataset"
    DEFAULT_GT_PATH = "/home/mrc689/Sampled_Dataset_GT/density_map"
    DEFAULT_LOG_PATH = "/home/mrc689/tf_logs"
    DEFAULT_RATIO_TRAINTEST_DATASET = 0.7
    DEFAULT_LEARNING_RATE = 0.00001
    DEFAULT_CHECKPOINT_PATH = "/home/mrc689/tf_ckpt"
    DEFAULT_LOG_FREQUENCY = 10

    # Create arguements to parse
    ap = argparse.ArgumentParser(description="Script to train the Convolutional neural network based FlowerCounter model in distributed GPU cluster.")

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
    proc = subprocess.Popen(['./gpu_profile'])
    print("start GPU profiling process with pid %s" % proc.pid)

    assign_tasks(args)
    duration = time.time() - start_time

    # Kill the GPU profiling processes.
    kill(proc.pid)

    print("Duration : ", duration)
