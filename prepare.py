import glob
import pickle
import itertools
import Model.config as config
import tensorflow as tf
import numpy as np
import cv2

def filter_item(directory):
    '''

    :param directory:
    :return: a list of filtered items
    '''

    cameraid_capturing_days=[]

    for filename in glob.glob(directory+"/*"):

        for each_image in glob.glob(filename+"/*"):
            image_name,camera_id = (each_image.split("/")[-1]).split("_")[0], (each_image.split("/")[-2])
            image_name = int(image_name.replace("frame",""))

            # if camera_id.split('-')[0] == '1109':
#
#                 dict= {}
#
#                 days_to_look_for = ['0706','0712']
#
#                 if camera_id.split('-')[1] in days_to_look_for:
#
#                     if (image_name >= 250 and image_name<= 1329):
#
#                         dict['camera_id'] = '1109'
#                         dict['capturing_day'] = camera_id.split('-')[1]
#                         dict['image_name'] = each_image
#                         cameraid_capturing_days.append(dict)

            if camera_id.split('-')[0] == '1237':

                dict = {}

                days_to_look_for = ['0715', '0720', '0725']

                if camera_id.split('-')[1] in days_to_look_for:

                    dict['camera_id'] = '1237'
                    dict['capturing_day'] = camera_id.split('-')[1]
                    dict['image_name'] = each_image

                    cameraid_capturing_days.append(dict)

    return cameraid_capturing_days


def get_filtered_dataset(img_dir,gt_dir):

    cameraid_capturing_days = filter_item(img_dir)

    annotated_gt = filter_item(gt_dir)

    return cameraid_capturing_days, annotated_gt


def find_them(img_list, gt_list):

    names_of_img_den_arr = []
    name_of_imgs = []
    culprits = []

    for each_item in gt_list:
        i_want_this_part = each_item['image_name'].split("/")[-1].split(".")[0]
        names_of_img_den_arr.append(i_want_this_part)

    for each_item in img_list:
        i_want_this_part = each_item['image_name'].split("/")[-1].split(".")[0]
        name_of_imgs.append(i_want_this_part)

    for i in name_of_imgs:
        if i not in names_of_img_den_arr:
            culprits.append(i)

    return culprits


def remove_missing_gt(img_list, gt_list):

    for each_item in img_list:

        if (each_item['image_name'].split("/")[-1].split(".")[0] in gt_list):
            img_list.remove(each_item)

    return img_list


def get_train_test_DataSet(image_path,gt_path,ratio):
    
    """
    Given input images and their ground truth, this function first filters unnecessary images ang gts. Then it checks if there are same number of input images 
    as the ground truths. If not then removes the not matching pairs and then sends us the training and testing set based on the ratio provided from the parameteres 
    being passed. 
      
    :param image_path: The directory that contains the input images. Images are expected to be stored in multiple directory.
    :param gt_path: The directory that contains the corresponding ground truth density map for input images.
    :return: lists of train and test dataset
    """
    
    filtered_image_dataset = []
    filtered_gt_dataset = []

    img_dataset, gt_labelset = get_filtered_dataset(image_path, gt_path)

    if (len(img_dataset) != len(gt_labelset)):

        ids = []

        for filename in glob.glob(gt_path + "/*"):
            ids.append(filename.split("/")[-1])


        for i in range(0, len(ids)):

            filtered_gt = list(
                filter(lambda x: x['camera_id'] == ids[i].split('-')[0] and x['capturing_day'] == ids[i].split('-')[1],
                       gt_labelset))
            filtered_img = list(
                filter(lambda x: x['camera_id'] == ids[i].split('-')[0] and x['capturing_day'] == ids[i].split('-')[1],
                       img_dataset))


            if (len(filtered_gt) != len(filtered_img)):

                msg = "Couldn't find the density maps for the following images for camera-id: {} , day: {}".format(
                    ids[i].split('-')[0], ids[i].split('-')[1])
                # print(msg)

                culprits = find_them(img_list=filtered_img, gt_list=filtered_gt)
                # print(culprits)

                refined_img_list = remove_missing_gt(img_list=filtered_img, gt_list=culprits)

                filtered_image_dataset.append(refined_img_list)
                filtered_gt_dataset.append(filtered_gt)

            else:

                filtered_image_dataset.append(filtered_img)
                filtered_gt_dataset.append(filtered_gt)

        # Flat the lists
        filtered_image_dataset = list(itertools.chain.from_iterable(filtered_image_dataset))
        filtered_gt_dataset = list(itertools.chain.from_iterable(filtered_gt_dataset))

    else:

        filtered_image_dataset = img_dataset
        filtered_gt_dataset = gt_labelset

    filtered_image_dataset = sorted(filtered_image_dataset, key=lambda k: k['image_name'])
    filtered_gt_dataset = sorted(filtered_gt_dataset, key=lambda k: k['image_name'])

    # Filtering only the image name from the each dictionary.
    filtered_image_dataset = list(map(lambda x: x['image_name'],filtered_image_dataset))
    filtered_gt_dataset = list(map(lambda x: x['image_name'],filtered_gt_dataset))

    trainset_limit = int(len(filtered_image_dataset) * ratio)
    
    train_img_set = filtered_image_dataset[:trainset_limit]
    train_gt_set = filtered_gt_dataset[:trainset_limit]
    test_img_set = filtered_image_dataset[trainset_limit:]
    test_gt_set = filtered_gt_dataset[trainset_limit:]
    
    return train_img_set,train_gt_set,test_img_set,test_gt_set


def read_npy_file(image_name,item):


    # The ground truth density map needs to be downsampled because after beign processed through the MAX-POOL layers the input is downsized in half for each MAX-POOL layer.
    data = np.load(item.decode())

    original_height = int(config.input_image_height)
    original_width =  int(config.input_image_width)
    width =  int(original_width/4)
    height = int(original_height/4)
    data = cv2.resize(data, (width, height))
    data = data * ((original_width * original_height) / (width * height))

    # !!!!!!!!!!!!!!!! This reshaping doesn't need to be done if the density map is multichanneled. !!!!!!!!!!!!!!!!!!!!!!
    data = np.reshape(data, [data.shape[1], data.shape[0], 1])
    return image_name,data.astype(np.float32)



def _parse_function(image_path,groundTruth_path):

    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=config.input_image_channels)
    image_normalized = tf.image.per_image_standardization(image_decoded)
    # Due to the variable size of input images, resizing was done to scale all images into a fix size.
    image_resized = tf.image.resize_images(image_normalized, [config.input_image_width, config.input_image_height])
    image = tf.cast(image_resized, tf.float32)

    return image,groundTruth_path
