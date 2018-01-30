# Installed modules
import numpy as np
# User-defined modules
from LearnAtariBoxing.config import *


# Data structure for implementing frame sequence
class FrameSequence:
    def __init__(self):
        self.reset()
    #end

    def insert(self, item):
        self.memory.append(item)
        if len(self.memory) > AGENT_HISTORY_LENGTH:
            del self.memory[0]
    #end

    def reset(self):
        ## Initialize by zero-filled frames
        self.memory = []
        for _ in range(AGENT_HISTORY_LENGTH):
            self.memory.append(np.zeros((PROCESSED_INPUT_WIDTH, PROCESSED_INPUT_HEIGHT)))
    #end

    ## Return memory as 1*M*N*length array
    def memory_as_array(self):
        return np.expand_dims(np.transpose(np.asarray(self.memory), (1, 2, 0)), axis=0)
    #end
#end


# Data structure for implementing replay memory
class ReplayMemory:
    def __init__(self):
        self.memory = []
    #end

    def insert(self, fs1, action, reward, fs2, done):
        self.memory.append({'fs1': fs1, 'action': action, 'reward': reward, 'fs2': fs2, 'done': done})
        if len(self.memory) > REPLAY_MEMORY_SIZE:
            del self.memory[0]
    #end

    def sample_mini_batch(self):
        return np.random.choice(self.memory, MINIBATCH_SIZE, replace=False)
    #end
#end