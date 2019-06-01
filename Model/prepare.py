import glob
import pickle
import itertools
import tensorflow as tf
import numpy as np
import cv2
import random
import collections
import Model.config as config

def CountFrequency(arr):

    freq = collections.Counter(arr)
    return freq	

def get_balanced_dataset(image,gt):

    number_of_bins = 6

    elements_in_bins =  [[] for _ in range(number_of_bins)]
 
    for i in range(0,len(image)):

        count = int(round(np.sum(np.load(gt[i]))))
        
        if (count == 0):
            elements_in_bins[0].append((image[i],gt[i]))

        if (count == 1):
            elements_in_bins[1].append((image[i],gt[i]))

        if (count == 2):
            elements_in_bins[2].append((image[i],gt[i]))
             
        if (count >= 3 and count <=4):
            elements_in_bins[3].append((image[i],gt[i]))

        if (count > 4 and count < 8):
            elements_in_bins[4].append((image[i],gt[i]))

        if (count >= 8):
            elements_in_bins[5].append((image[i],gt[i]))

    print("Number of Bins : ",len(elements_in_bins),"\n")
    print("Images per Bin: ",list(map(lambda x : len(x),elements_in_bins)))
    
    randomly_sampled_dataset = []

    # Take either the bin which has the minimum number of flowers.
    # Or, take the last bin containing images which has maximum number of flowers.
    number_of_samples = np.min(list(map(lambda x : len(x),elements_in_bins)))
    print("Number of samples to be collected from each bin : ",number_of_samples)

    random.seed(354)

    for i in range(0,len(elements_in_bins)):
        randomly_sampled_dataset.append(random.sample(elements_in_bins[i],number_of_samples))

    flatten_list = list(itertools.chain.from_iterable(randomly_sampled_dataset))

    random.shuffle(flatten_list)
 
    return flatten_list


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
            
            if camera_id.split('-')[0] == '1109':

                 dict= {}

                 #days_to_look_for = ['0704','0710']
                 days_to_look_for = ['0704']
  
                 if camera_id.split('-')[1] in days_to_look_for:

                     dict['camera_id'] = '1109'
                     dict['capturing_day'] = camera_id.split('-')[1]
                     dict['image_name'] = each_image
                     cameraid_capturing_days.append(dict)

    return cameraid_capturing_days


def get_filtered_dataset(img_dir,gt_dir):

    cameraid_capturing_days = filter_item(img_dir)

    annotated_gt = filter_item(gt_dir)

    return cameraid_capturing_days, annotated_gt


def find_them(img_list, gt_list):

    names_of_gt = []
    names_of_imgs = []
    culprits = []
    present_in_gt_but_not_in_imgs = []

    for each_item in gt_list:
        name_of_the_file = each_item['image_name'].split("/")[-1].split(".")[0]
        names_of_gt.append(name_of_the_file)

    for each_item in img_list:
        name_of_the_file = each_item['image_name'].split("/")[-1].split(".")[0]
        names_of_imgs.append(name_of_the_file)

    # If the gt list is smaller than the image list, then remove the unmatching images. 
    if (len(img_list) > len(gt_list)):

        for i in names_of_imgs:
            if i not in names_of_gt:
                culprits.append(i)

        for each_element in names_of_gt:
            found = False

            for each_line in names_of_imgs:

                if str(each_element) in str(each_line):
                    found = True

            if not found:
                print("Couldn't find the images for the ground truth :", each_element)

    else:

        for i in names_of_gt:
            if i not in names_of_imgs:
                culprits.append(i)

    return culprits


def remove_missing_img_gt(item_list, delete_from_this_list):

    for i in item_list:
        for j in delete_from_this_list:

            if (j['image_name'].split("/")[-1].split(".")[0] == i):
                delete_from_this_list.remove(j)

    return delete_from_this_list


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
                print(msg)

                culprits = find_them(img_list=filtered_img, gt_list=filtered_gt)
                print("Length of culprits", len(culprits))

                if (len(filtered_img) > len(filtered_gt)):
                    print('Before removing images with abscent gts : ', len(filtered_img))
                    refined_img_list = remove_missing_img_gt(item_list = culprits, delete_from_this_list = filtered_img)
                    refined_gt_list = filtered_gt
                    print('After removing images with abscent gts : ', len(refined_img_list))
                else:

                    refined_img_list = filtered_img
                    refined_gt_list = remove_missing_img_gt(item_list = culprits, delete_from_this_list = filtered_gt)

                print("Length of img list and gt list", len(refined_img_list),len(refined_gt_list))

                filtered_image_dataset.append(refined_img_list)
                filtered_gt_dataset.append(refined_gt_list)

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

    ##################################################
   
    # If manual annotation is used then uncomment the following line.
    dataset = get_balanced_dataset(filtered_image_dataset,filtered_gt_dataset) 
    #dataset = get_balanced_dataset_automatic(filtered_image_dataset,filtered_gt_dataset) 

    filtered_image_dataset = []
    filtered_gt_dataset = []

    for index in range(0,len(dataset)):

        filtered_image_dataset.append(dataset[index][0])
        filtered_gt_dataset.append(dataset[index][1])

    '''
    # For experimental purpose added few manually annotated images from 1109-0710. /home/mrc689/results/manual/1109-0710

    with open('/home/mrc689/results/manual/1109-0710/shuffled_test_img.txt',"r") as file_obj:

        lines = file_obj.readlines()

        for eachline in lines:
            eachline = eachline.strip()
            filtered_image_dataset.append(eachline)

    with open('/home/mrc689/results/manual/1109-0710/shuffled_test_gt.txt',"r") as file_obj:

        lines = file_obj.readlines()

        for eachline in lines:
            eachline = eachline.strip()
            filtered_gt_dataset.append(eachline)
    '''

    filtered_image_dataset = sorted(filtered_image_dataset)
    filtered_gt_dataset = sorted(filtered_gt_dataset)

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
    # Removed the downsampling because in the modified version I have used deconvolution layers twice.
    # width =  int(original_width/4)
    # height = int(original_height/4)
    width =  int(original_width)
    height = int(original_height)
    
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

    return image_path,image,groundTruth_path



if __name__ == "__main__":

   image_path = "/home/mrc689/Sampled_Dataset"
   gt_path = "/home/mrc689/Sampled_Dataset_GT/density_map/manual"
   dataset_train_test_ratio = 0.7

   img , gt, dhiki, chiki = get_train_test_DataSet(image_path,gt_path,dataset_train_test_ratio)

   print(img[151],gt[151])
   print(img[-10],gt[-10])
   print(chiki[151],dhiki[151])
   print(chiki[-10],dhiki[-10])
    
