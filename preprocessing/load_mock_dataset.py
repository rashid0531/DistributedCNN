import glob
import pathlib


def load_training_image_groundtruth(image_dir, gt_dir):
    training_imgs_path = list(image_dir.glob('*'))
    training_gt_path = list(gt_dir.glob('*'))
    return training_imgs_path, training_gt_path
