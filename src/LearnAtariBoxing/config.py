# Macro size of the processed image
PROCESSED_INPUT_WIDTH = 84
PROCESSED_INPUT_HEIGHT = 84

# Minibatch size
MINIBATCH_SIZE = 32

# The number of training per one minibatch
NUM_TRAIN_PER_MINIBATCH = 1

# Replay memory size
REPLAY_MEMORY_SIZE = 250000

REPLAY_START_SIZE = 1000 #32

# Agent history length
AGENT_HISTORY_LENGTH = 4

# Discount factor for Q-learning update
DISCOUNT_FACTOR = 0.99

# Target network update frequency
TARGET_UPDATE_FREQUENCY = 10000

# It determines the epsilon decay rate
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000

# Epsilon decay frequency
EPSILON_DECAY_FREQUENCY = 1