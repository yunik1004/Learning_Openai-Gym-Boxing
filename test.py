# Python batteries
import os
from multiprocessing import Process
# Installed modules
from gym.wrappers import Monitor
# User-defined modules
from agents import Agent_Atari
from tools import current_time_microsecond


# Test the agent program
def test(env, dir_model, dir_save, num_episodes):
    if not os.path.isdir(dir_model):
        raise OSError(2, 'No such directory', dir_model)
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    agent = Agent_Atari(env=env, exploration=False)

    # Save the processes
    procs = []

    for itr_ep in range(num_episodes):
        dir_record = os.path.join(dir_save, 'record', 'test-ep_%d' % (itr_ep, ))
        proc = Process(target=test_one, args=(env, dir_record, agent))
        procs.append(proc)
        proc.start()
    #end

    for proc in procs:
        proc.join()
    #end
#end

# Test one episode
def test_one(env, dir_record, agent):
    env_record = Monitor(env, directory=dir_record, force=True)
    ob = env_record.reset()
    env_record.seed(current_time_microsecond())
    while True:
        action = agent.next_action(ob)
        ob, reward, done, _ = env_record.step(action)
        if done:
            break
    #end
#end