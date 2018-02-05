# Python batteries
import json, os
from collections import OrderedDict
# Installed modules
import numpy as np
from gym.wrappers import Monitor
# User-defined modules
from LearnAtariBoxing.agents import Agent_Atari
from LearnAtariBoxing.preprocessors import atari_img_preprocess


# Test the agent program
def test(env, path_model, dir_save, num_episodes):
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    agent = Agent_Atari(env=env, exploration=0)
    agent.onlineDQN.import_model(path_model)

    list_rewards =[]
    for itr_ep in range(num_episodes):
        dir_record = os.path.join(dir_save, 'records', 'test-ep_%d' % (itr_ep, ))
        total_reward = test_one(agent, dir_record, itr_ep)
        list_rewards.append(total_reward)
    #end

    # Export the results
    results = OrderedDict()
    results['model_location'] = path_model
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
def test_one(agent, dir_record, itr):
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
        agent.replay_memory.insert(fs1, action, np.clip(reward, -1, 1), fs2, done)
        if done:
            break
    #end

    total_reward = env_record.get_episode_rewards()[0]
    
    env_record.close()

    return total_reward
#end