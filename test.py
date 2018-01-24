# Python batteries
import os
# User-defined modules
from agents import Agent_Atari


# Test the agent program
def test(env, dir_model, dir_save):
    if not os.path.isdir(dir_model):
        raise OSError(2, 'No such directory', dir_model)
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    dqn = Agent_Atari()

    ob = env.reset()
    while True:
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        if done:
            break
    #end
#end