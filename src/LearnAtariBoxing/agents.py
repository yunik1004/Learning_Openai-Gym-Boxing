# Installed modules
import numpy as np
import tensorflow as tf
# User-defined modules
from LearnAtariBoxing.data_structures import ReplayMemory
from LearnAtariBoxing.config import *
from LearnAtariBoxing.preprocessors import atari_img_preprocess


# Learning agent for Atari games
# # Parameters:
#  - exploration: boolean. If true, do the exploration.
class Agent_Atari:
    def __init__(self, env, exploration, **kwargs):
        self.env = env
        self.exploration = exploration
        self.reset(**kwargs)
    #end

    def reset(self, size_replay=4, update_frequency=4):
        self.size_replay = size_replay
        self.replay_memory = ReplayMemory(self.size_replay)
        self.update_frequency = update_frequency

        ### Create session
        self.close_session()
        self.sess = tf.Session()

        ### Generate two DQNs - online, target
        self.onlineDQN = DQN(self.sess)
        self.targetDQN = DQN(self.sess)

        ### Initialize all tensorflow variables
        self.sess.run(tf.global_variables_initializer())
        self.targetDQN.update_variables(self.onlineDQN)
    #end

    ## Close the tensorflow session
    def close_session(self):
        try:
            self.sess.close()
        except:
            pass
    #end

    ## The shape of the observation(=observed image) should be M*N*3
    def next_action(self, observation):
        self.replay_memory.insert(atari_img_preprocess(observation))
        return self.env.action_space.sample()
    #end
#end


# Deep Q-Learning Network
class DQN:
    def __init__(self, sess, size_replay=4, num_actions=14):
        ## Tensorflow session
        self.sess = sess

        ## Network features
        self.num_feature_map1 = 32
        self.num_feature_map2 = 64
        self.num_neuron_unit = 512
        self.dropout = 0.75
        self.size_replay = size_replay
        self.num_actions = num_actions

        ## Input, output tensor
        input_x = tf.placeholder(tf.float32, [None, PROCESSED_INPUT_WIDTH, PROCESSED_INPUT_HEIGHT, self.size_replay])
        self.action = tf.placeholder(tf.float32, [None, num_actions])
        self.reward = tf.placeholder(tf.float32, [None, 1])

        ## Dropout probability
        keep_prob = tf.placeholder(tf.float32)

        ## First convolutional layer
        self.conv1 = ConvolutionalLayer(size_filter=8, depth_input=self.size_replay, num_feature_map=self.num_feature_map1, stride=4)
        output_conv1 = self.conv1.output(input_x)
        output_conv1 = tf.nn.dropout(output_conv1, keep_prob)
        size_output_conv1 = self.conv1.size_output(PROCESSED_INPUT_WIDTH)

        ## Second convolutional layer
        self.conv2 = ConvolutionalLayer(size_filter=4, depth_input=self.num_feature_map1, num_feature_map=self.num_feature_map2, stride=2)
        output_conv2 = self.conv2.output(output_conv1)
        output_conv2 = tf.nn.dropout(output_conv2, keep_prob)
        size_output_conv2 = self.conv2.size_output(size_output_conv1)

        ## Reshaping output
        num_element_output_conv2 = size_output_conv2 * size_output_conv2 * self.num_feature_map2
        output_conv2_reshaped = tf.reshape(output_conv2, [-1, num_element_output_conv2])

        ## Fully connected layer
        self.wd1 = tf.Variable(tf.truncated_normal([num_element_output_conv2, self.num_neuron_unit], stddev=0.1))
        self.bd1 = tf.Variable(tf.truncated_normal([self.num_neuron_unit], stddev=0.1))
        dense1 = tf.nn.relu(tf.add(tf.matmul(output_conv2_reshaped, self.wd1), self.bd1))
        dense1 = tf.nn.dropout(dense1, keep_prob)

        ## Output layer
        self.wout = tf.Variable(tf.truncated_normal([self.num_neuron_unit, self.num_actions], stddev=0.1))
        self.bout = tf.Variable(tf.truncated_normal([self.num_actions], stddev=0.1))
        self.pred = tf.add(tf.matmul(dense1, self.wout), self.bout)

        ## Cost function and optimizer
    #end

    def update_variables(self, dqn):
        self.sess.run(tf.assign(self.conv1.wc, dqn.conv1.wc))
        self.sess.run(tf.assign(self.conv1.bc, dqn.conv1.bc))
        self.sess.run(tf.assign(self.conv2.wc, dqn.conv2.wc))
        self.sess.run(tf.assign(self.conv2.bc, dqn.conv2.bc))
        self.sess.run(tf.assign(self.wd1, dqn.wd1))
        self.sess.run(tf.assign(self.bd1, dqn.bd1))
        self.sess.run(tf.assign(self.wout, dqn.wout))
        self.sess.run(tf.assign(self.bout, dqn.bout))
    #end
#end


# Implementation of convolutional layer
class ConvolutionalLayer:
    def __init__(self, size_filter, depth_input, num_feature_map, stride):
        self.size_filter = size_filter
        self.depth_input = depth_input
        self.num_feature_map = num_feature_map
        self.stride = stride

        self.wc = tf.Variable(tf.truncated_normal([self.size_filter, self.size_filter, self.depth_input, self.num_feature_map], stddev=0.1))
        self.bc = tf.Variable(tf.truncated_normal([self.num_feature_map], stddev=0.1))
    #end

    def output(self, input_x):
        return self.conv2d(input_x, self.wc, self.bc, self.stride)
    #end

    def size_output(self, size_input):
        return (size_input - self.size_filter) / self.stride + 1
    #end

    def conv2d(self, x, w, b, stride):
        return tf.nn.relu(tf.nn.bias_add(
                            tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding='VALID'),
                            b))
    #end
#end