from pathlib import Path
from preprocessing.load_mock_dataset import load_training_image_groundtruth

DEFAULT_IMAGE_PATH = Path('./mock_data/training_images/1109-0704/')
DEFAULT_GT_PATH = Path('./mock_data/ground_truth/1109-0704/')

train_images, train_gts = load_training_image_groundtruth(DEFAULT_IMAGE_PATH, DEFAULT_GT_PATH)
