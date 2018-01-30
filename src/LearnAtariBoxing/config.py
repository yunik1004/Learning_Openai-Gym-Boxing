# The number of atari action
NUM_ACTIONS = 18

# Macro size of the processed image
PROCESSED_INPUT_WIDTH = 84
PROCESSED_INPUT_HEIGHT = 84

# Minibatch size
MINIBATCH_SIZE = 32

# Replay memory size
REPLAY_MEMORY_SIZE = 1000000

# Agent history length
AGENT_HISTORY_LENGTH = 4

# Discount factor for Q-learning update
DISCOUNT_FACTOR = 0.99

# Target network update frequency
TARGET_UPDATE_FREQUENCY = 10000

# Epsilon decay for exploitation
EPSILON_DECAY = 0.99