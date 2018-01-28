# Installed modules
import numpy as np
import tensorflow as tf
# User-defined modules
from preprocessors import atari_img_preprocess


# Learning agent for Atari games
# # Parameters:
#  - exploration: boolean. If true, do the exploration.
class Agent_Atari:
    def __init__(self, env, exploration):
        self.env = env
        self.exploration = exploration
        self.reset()
    #end

    def reset(self):
        pass
    #end

    def next_action(self, observation):
        return self.env.action_space.sample()
    #end
#end