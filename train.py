# Python batteries
import os
# User-defined modules
from preprocessors import atari_img_preprocess


# Train the agent program
def train(env, dir_save):
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)

    ob = env.reset()
    while True:
        ob_preprocessed = atari_img_preprocess(ob)
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        if done:
            break
    #end
#end