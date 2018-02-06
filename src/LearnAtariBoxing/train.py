# Python batteries
import os
# Installed modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import Monitor
# User-defined modules
from LearnAtariBoxing.agents import Agent_Atari
from LearnAtariBoxing.config import *
from LearnAtariBoxing.preprocessors import atari_img_preprocess


# Train the agent program
def train(env, dir_save, num_episodes):
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    agent = Agent_Atari(env=env, exploration=1)

    list_rewards = []
    list_costs = []
    list_errors = []

    for itr_ep in range(num_episodes):
        dir_record = os.path.join(dir_save, 'records', 'train-ep_%d' % (itr_ep, ))
        total_reward = train_one(agent, dir_record)
        list_rewards.append(total_reward)
        list_costs += agent.costs
        list_errors += agent.errors
        agent.reset_costs()
        agent.reset_errors()
    #end

    ## Save model
    agent.save_model(os.path.join(dir_save, 'model.ckpt'))

    ## Save cost graph per iteration
    costs_episode = zip(*list_costs)
    fig = plt.figure()
    plt.plot(*costs_episode)
    plt.title('Costs during training the agent')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    fig.savefig(os.path.join(dir_save, 'costs.png'))
    plt.close(fig)

    ## Save error graph per iteration
    errors_episode = zip(*list_errors)
    fig = plt.figure()
    plt.plot(*errors_episode)
    plt.title('Errors during training the agent')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    fig.savefig(os.path.join(dir_save, 'errors.png'))
    plt.close(fig)

    ## Save total rewards graph
    fig = plt.figure()
    plt.plot(range(num_episodes), list_rewards)
    plt.title('Total rewards during training the agent')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    fig.savefig(os.path.join(dir_save, 'total_rewards.png'))
    plt.close(fig)

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
        if len(agent.replay_memory.memory) >= REPLAY_START_SIZE:
            agent.learn()
        ## If done == True, then this game is finished
        if done:
            break
    #end

    ## Save the model
    agent.save_model(os.path.join(dir_record, 'model.ckpt'))

    total_reward = env_record.get_episode_rewards()[0]

    env_record.close()

    ## Save cost graph per iteration
    costs_episode = zip(*agent.costs)
    fig = plt.figure()
    plt.plot(*costs_episode)
    plt.title('Costs during training the agent')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    fig.savefig(os.path.join(dir_record, 'costs.png'))
    plt.close(fig)

    ## Save error graph per iteration
    errors_episode = zip(*agent.errors)
    fig = plt.figure()
    plt.plot(*errors_episode)
    plt.title('Errors during training the agent')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    fig.savefig(os.path.join(dir_record, 'errors.png'))
    plt.close(fig)

    return total_reward
#end