#!/usr/bin/env python

# Python batteries
import argparse, os
from datetime import datetime
# Installed modules
import gym
# User-defined modules
from test import test
from train import train


# Arguments
parser = argparse.ArgumentParser(description="Training/testing Atari 'Boxing' game")
group_task = parser.add_mutually_exclusive_group()
group_task.add_argument("--test", action="store_true", help="Test the agent")
group_task.add_argument("--train", action="store_true", help="Train the agent")
parser.add_argument("-m", "--model", type=str, help="Location of the directory where the trained model files are located. In the case of training session, this argument is meaningless. In the case of testing session, this argument must be required.")
parser.add_argument("-s", "--save", type=str, help="Location of the directory for saving the result. In the case of training session, the default location is '%s'. In the case of testing session, the default location is '%s'." % (os.path.join('<project_directory>', 'models', '<current_timestamp>'), os.path.join('<project_directory>', 'test_results', '<current_timestamp>')))
parser.add_argument("-e", "--episodes", type=int, help="The number of episodes. It should be a positive integer.", required=True)


# Main function
def main():
    args = parser.parse_args()

    if args.train is None and args.test is None:
        parser.error("One of the '--test' or '--train' arguments should be required")

    dir_save = args.save
    if dir_save:
        dir_save = os.path.realpath(dir_save)
    elif args.train:
        dir_save = os.path.join(realpath_thisfile(), 'models', current_timestamp())
    elif args.test:
        dir_save = os.path.join(realpath_thisfile(), 'test_results', current_timestamp())

    dir_model = args.model
    if dir_model:
        dir_model = os.path.realpath(dir_model)
    elif args.test:
        parser.error("'--model' argument is required for testing session")

    num_episodes = args.episodes
    if num_episodes <= 0:
        parser.error("The number of episodes should be a positive integer")

    env = gym.make('Boxing-v0')
    env.seed(0)

    if args.train:
        train(env, dir_save, num_episodes)
    elif args.test:
        test(env, dir_model, dir_save, num_episodes)

    env.close()
#end

# Generate the current timestamp
def current_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#end

# Get the real path of this file
def realpath_thisfile():
    return os.path.dirname(os.path.realpath(__file__))
#end


# Run main function
if __name__ == "__main__":
    main()
#end