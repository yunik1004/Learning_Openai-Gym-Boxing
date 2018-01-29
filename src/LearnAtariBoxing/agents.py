# Installed modules
import numpy as np
import tensorflow as tf
# User-defined modules
from data_structures import ReplayMemory
from LearnAtariBoxing.config import *
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
        self.replay_memory = ReplayMemory(1)
    #end

    ## The shape of the observation(=observed image) should be M*N*3
    def next_action(self, observation):
        self.replay_memory.insert(atari_img_preprocess(observation))
        return self.env.action_space.sample()
    #end
#end