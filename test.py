import os

import numpy as np
import tensorflow as tf


# Test the agent program
def test(env, dir_model, dir_save):
    env.reset()

    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    #end
#end