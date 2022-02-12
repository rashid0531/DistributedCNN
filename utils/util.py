from PIL import Image, ImageFile
import numpy as np

def read_image_using_PIL(image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(image)
    image = image.resize((224,224))
    image = np.asarray(image, np.uint8)
    return image