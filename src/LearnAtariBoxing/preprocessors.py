# Installed modules
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
# User-defined modules
from LearnAtariBoxing.config import *


# Preprocess the atari game image to gray-scale 84*84 image
def atari_img_preprocess(img):
    return resize(rgb2gray(img), (PROCESSED_INPUT_WIDTH, PROCESSED_INPUT_HEIGHT), mode='constant')
#end