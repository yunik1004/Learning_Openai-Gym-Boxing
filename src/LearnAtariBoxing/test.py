# Python batteries
import json, multiprocessing, os
from collections import OrderedDict
# Installed modules
import numpy as np
from gym.wrappers import Monitor
# User-defined modules
from LearnAtariBoxing.agents import Agent_Atari
from LearnAtariBoxing.preprocessors import atari_img_preprocess


# Test the agent program
def test(env, dir_model, dir_save, num_episodes):
    if not os.path.isdir(dir_model):
        raise OSError(2, 'No such directory', dir_model)
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    agent = Agent_Atari(env=env, exploration=False)

    # For multiprocessing
    manager = multiprocessing.Manager()
    list_rewards = manager.list([None]*num_episodes)
    procs = []

    for itr_ep in range(num_episodes):
        dir_record = os.path.join(dir_save, 'records', 'test-ep_%d' % (itr_ep, ))
        proc = multiprocessing.Process(target=test_one, args=(agent, dir_record, itr_ep, list_rewards))
        procs.append(proc)
        proc.start()
    #end

    for proc in procs:
        proc.join()
    #end

    list_rewards = list(list_rewards)

    # Export the results
    results = OrderedDict()
    results['model_location'] = dir_model
    results['mean'] = np.mean(list_rewards)
    results['std'] = np.std(list_rewards)
    results['median'] = np.median(list_rewards)
    results['total_rewards'] = list_rewards

    with open(os.path.join(dir_save, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    #end

    agent.close_session()
#end

# Test one episode
def test_one(agent, dir_record, itr, list_rewards):
    agent.env.seed(itr)

    env_record = Monitor(agent.env, directory=dir_record)

    ob = env_record.reset()
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
        if done:
            break
    #end

    list_rewards[itr] = env_record.get_episode_rewards()[0]
    env_record.close()
#end