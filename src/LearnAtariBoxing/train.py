# Python batteries
import os
# Installed modules
from gym.wrappers import Monitor
# User-defined modules
from LearnAtariBoxing.agents import Agent_Atari
from LearnAtariBoxing.config import *
from LearnAtariBoxing.preprocessors import atari_img_preprocess


# Train the agent program
def train(env, dir_save, num_episodes):
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    agent = Agent_Atari(env=env, exploration=0.9)

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
    agent.frame_sequence.reset()
    agent.frame_sequence.insert(atari_img_preprocess(ob))
    while True:
        fs1 = agent.frame_sequence.memory_as_array()
        ## Find next action
        action = agent.next_action()
        ob, reward, done, _ = env_record.step(action)
        agent.frame_sequence.insert(atari_img_preprocess(ob))
        fs2 = agent.frame_sequence.memory_as_array()
        ## Save results into the replay memory
        agent.replay_memory.insert(fs1, action, reward, fs2, done)
        ## Perform learning
        if len(agent.replay_memory.memory) >= MINIBATCH_SIZE:
            agent.learn()
        ## If done == True, then this game is finished
        if done:
            break
    #end

    env_record.close()
#end