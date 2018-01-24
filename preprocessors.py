# Installed modules
import numpy as np
from skimage.transform import resize


def atari_img_preprocess(img):
    return resize(img_to_gray(img), (84, 84), mode='constant')
#end

def img_to_gray(img):
    return np.dot(img, [0.299, 0.587, 0.114])
#end