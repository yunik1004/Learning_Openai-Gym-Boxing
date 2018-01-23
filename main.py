#!/usr/bin/env python

import gym


# Main function
def main():
    env = gym.make('Boxing-v0')

    env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    #end
#end


# Run main function
if __name__ == "__main__":
    main()
#end