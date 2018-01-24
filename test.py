import os

import numpy as np
import tensorflow as tf


# Test the agent program
def test(env, dir_model, dir_save):
    if not os.path.isdir(dir_model):
        raise OSError(2, 'No such directory', dir_model)
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    env.reset()

    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    #end
#end