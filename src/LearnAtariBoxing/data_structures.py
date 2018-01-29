# Installed modules
import numpy as np
# User-defined modules
from LearnAtariBoxing.config import *


# Data structure for implementing replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity

        ## Initialize by zero-filled frames
        for _ in range(self.capacity):
            self.memory.append(np.zeros((PROCESSED_INPUT_WIDTH, PROCESSED_INPUT_HEIGHT)))
    #end

    def insert(self, item):
        self.memory.append(item)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        return
    #end

    ## Return memory as 1*M*N*capacity array
    def memory_as_array(self):
        return np.expand_dims(np.transpose(np.asarray(self.memory), (1, 2, 0)), axis=0)
    #end
#end