# Macro size of the processed image
PROCESSED_INPUT_WIDTH = 28
PROCESSED_INPUT_HEIGHT = 28

# Minibatch size
MINIBATCH_SIZE = 64

# The number of training per one minibatch
NUM_TRAIN_PER_MINIBATCH = 1

# Replay memory size
REPLAY_MEMORY_SIZE = 250000

# Agent history length
AGENT_HISTORY_LENGTH = 4

# Discount factor for Q-learning update
DISCOUNT_FACTOR = 0.99

# Target network update frequency
TARGET_UPDATE_FREQUENCY = 10000

# Epsilon decay rate for exploitation
EPSILON_DECAY_RATE = 0.99

# Epsilon decay frequency
EPSILON_DECAY_FREQUENCY = 10000