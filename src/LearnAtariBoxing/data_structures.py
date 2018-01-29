# Data structure for implementing experience memory
class ExperienceMemory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
    #end

    def insert(self, item):
        self.memory.append(item)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        return
    #end
#end