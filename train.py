# Python batteries
import os
# Installed modules
from gym.wrappers import Monitor
# User-defined modules
from agents import Agent_Atari


# Train the agent program
def train(env, dir_save, num_episodes):
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    agent = Agent_Atari(env=env, exploration=True)

    for itr_ep in range(num_episodes):
        env_record = Monitor(env, directory=os.path.join(dir_save, 'record', 'train-ep_%d' % (itr_ep, )), force=True)
        train_one(env_record, dir_save, agent)
    #end
#end

# Train one episode
def train_one(env, dir_save, agent):
    ob = env.reset()
    while True:
        action = agent.next_action(ob)
        ob, reward, done, _ = env.step(action)
        if done:
            break
    #end
#end