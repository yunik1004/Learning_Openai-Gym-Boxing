# Python batteries
import os
# Installed modules
from gym.wrappers import Monitor
# User-defined modules
from LearnAtariBoxing.agents import Agent_Atari


# Train the agent program
def train(env, dir_save, num_episodes):
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    agent = Agent_Atari(env=env, exploration=True)

    for itr_ep in range(num_episodes):
        dir_record = os.path.join(dir_save, 'records', 'train-ep_%d' % (itr_ep, ))
        train_one(agent, dir_record)
    #end

    agent.save_model(os.path.join(dir_save, 'model.ckpt'))
    agent.close_session()
#end

# Train one episode
def train_one(agent, dir_record, seed=None):
    if not seed is None:
        agent.env.seed(seed)

    env_record = Monitor(agent.env, directory=dir_record)

    ob = env_record.reset()
    while True:
        action = agent.next_action(ob)
        ob, reward, done, _ = env_record.step(action)
        if done:
            break
    #end

    env_record.close()
#end