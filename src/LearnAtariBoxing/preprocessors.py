# Installed modules
import numpy as np
from skimage.transform import resize
# User-defined modules
from LearnAtariBoxing.config import *


# Preprocess the atari game image to gray-scale 84*84 image
def atari_img_preprocess(img):
    return resize(img_to_gray(img), (PROCESSED_INPUT_WIDTH, PROCESSED_INPUT_HEIGHT), mode='constant')
#end

# Modify the image into gray-scale image
## The shape of the image should be M*N*3
def img_to_gray(img):
    return np.dot(img, [0.299, 0.587, 0.114])
#end