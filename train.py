import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Train the agent program
def train(env, dir_save):
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    env.reset()

    print('train')
#end