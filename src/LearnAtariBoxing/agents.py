# Installed modules
import numpy as np
import tensorflow as tf
# User-defined modules
from LearnAtariBoxing.data_structures import FrameSequence, ReplayMemory
from LearnAtariBoxing.config import *


# Learning agent for Atari games
class Agent_Atari:
    def __init__(self, env, exploration, **kwargs):
        self.env = env
        self.init_exploration = exploration
        self.final_exploration = 0.1
        self.reset(**kwargs)
    #end

    def reset(self, update_frequency=4):
        self.replay_memory = ReplayMemory()
        self.frame_sequence = FrameSequence()
        self.update_frequency = update_frequency
        self.current_exploration = self.init_exploration
        self.num_online_updated = 0
        self.reset_costs()
        self.reset_errors()

        ### Create session
        self.close_session()
        self.sess = tf.Session()

        ### Generate two DQNs - online, target
        self.onlineDQN = DQN(self.sess, self.env)
        self.targetDQN = DQN(self.sess, self.env)

        ### Initialize all tensorflow variables
        self.sess.run(tf.global_variables_initializer())
        self.targetDQN.update_variables(self.onlineDQN)
    #end

    ## Reset the frame sequence - Will be called when the new game starts
    def reset_frame_sequence(self):
        self.frame_sequence.reset()
    #end

    def reset_costs(self):
        self.costs = [] # [(itr, cost), ...]
    #end

    def reset_errors(self):
        self.errors = [] # [(itr, error), ...]
    #end

    ## Save the variables of onlineDQN into file
    def save_model(self, file_path):
        self.onlineDQN.save_model(file_path)
    #end

    ## Close the tensorflow session
    def close_session(self):
        try:
            self.sess.close()
        except:
            pass
    #end

    ## Return next action
    def next_action(self):
        # Exploration
        if np.random.sample() < self.current_exploration:
            action =  self.env.action_space.sample()
        # Exploitation
        else:
            qvalues = self.onlineDQN.output(self.frame_sequence.memory_as_array())
            action = np.argmax(qvalues)
        return action
    #end

    def learn(self):
        mini_batches = self.replay_memory.sample_mini_batch()
        for _ in range(NUM_TRAIN_PER_MINIBATCH):
            for batch in mini_batches:
                reward = batch['reward']
                if not batch['done']:
                    reward += DISCOUNT_FACTOR * np.argmax(self.targetDQN.output(batch['fs1']))
                current_cost = self.onlineDQN.current_cost(batch['fs1'], batch['action'], reward)
                current_error = self.onlineDQN.current_error(batch['fs1'], batch['action'], reward)
                self.costs.append((self.num_online_updated, current_cost))
                self.errors.append((self.num_online_updated, current_error))
                self.onlineDQN.optimize(batch['fs1'], batch['action'], reward)
                self.num_online_updated += 1
                ## Decreasing exploitation rate
                if self.num_online_updated % EPSILON_DECAY_FREQUENCY == 0:
                    self.current_exploration = min(max(self.current_exploration - EPSILON_DECAY_RATE, self.final_exploration), self.current_exploration)
                if self.num_online_updated % TARGET_UPDATE_FREQUENCY == 0:
                    self.update_targetDQN()
            #end
        #end
    #end

    ## Update target network
    def update_targetDQN(self):
        self.targetDQN.update_variables(self.onlineDQN)
    #end
#end


# Deep Q-Learning Network
class DQN:
    def __init__(self, sess, env):
        ## Tensorflow session
        self.sess = sess

        ## Network features
        self.num_feature_map1 = 16
        self.num_feature_map2 = 32
        self.num_neuron_unit = 256
        self.dropout = 0.75

        ## Input, output tensor
        self.input_x = tf.placeholder(tf.float32, [None, PROCESSED_INPUT_WIDTH, PROCESSED_INPUT_HEIGHT, AGENT_HISTORY_LENGTH])
        self.action = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        ## Dropout probability
        keep_prob = tf.constant(self.dropout)

        ## First convolutional layer
        self.conv1 = ConvolutionalLayer(size_filter=8, depth_input=AGENT_HISTORY_LENGTH, num_feature_map=self.num_feature_map1, stride=4)
        output_conv1 = self.conv1.output(self.input_x)
        #output_conv1 = tf.nn.dropout(output_conv1, keep_prob)
        size_output_conv1 = self.conv1.size_output(PROCESSED_INPUT_WIDTH)

        ## Second convolutional layer
        self.conv2 = ConvolutionalLayer(size_filter=4, depth_input=self.num_feature_map1, num_feature_map=self.num_feature_map2, stride=2)
        output_conv2 = self.conv2.output(output_conv1)
        #output_conv2 = tf.nn.dropout(output_conv2, keep_prob)
        size_output_conv2 = self.conv2.size_output(size_output_conv1)

        ## Reshaping output
        num_element_output_conv2 = size_output_conv2 * size_output_conv2 * self.num_feature_map2
        output_conv2_reshaped = tf.reshape(output_conv2, [-1, num_element_output_conv2])

        ## Fully connected layer
        self.wd1 = tf.Variable(tf.truncated_normal([num_element_output_conv2, self.num_neuron_unit], stddev=0.1))
        self.bd1 = tf.Variable(tf.truncated_normal([self.num_neuron_unit], stddev=0.1))
        dense1 = tf.nn.relu(tf.add(tf.matmul(output_conv2_reshaped, self.wd1), self.bd1))
        #dense1 = tf.nn.dropout(dense1, keep_prob)

        ## Output layer
        self.wout = tf.Variable(tf.truncated_normal([self.num_neuron_unit, env.action_space.n], stddev=0.1))
        self.bout = tf.Variable(tf.truncated_normal([env.action_space.n], stddev=0.1))
        self.pred = tf.add(tf.matmul(dense1, self.wout), self.bout)

        ## Cost function and optimizer - tf.gather(self.pred, self.action)
        qvalue = tf.reduce_sum(tf.multiply(self.pred, tf.one_hot(self.action, env.action_space.n)))
        self.error = self.reward - qvalue
        self.cost = tf.reduce_mean(tf.square(tf.clip_by_value(self.error, -1, 1)))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95)
        self.training_step = optimizer.minimize(self.cost)

        ## Saver setting
        self.saver = tf.train.Saver({self.conv1.wc, self.conv1.bc, self.conv2.wc, self.conv2.bc, self.wd1, self.bd1, self.wout, self.bout}, max_to_keep=0)
    #end

    ## Run the optimizer
    def optimize(self, input_x, action, reward):
        self.sess.run(self.training_step, feed_dict={self.input_x: input_x, self.action: action, self.reward: reward})
    #end

    ## Return the current cost
    def current_cost(self, input_x, action, reward):
        return self.sess.run(self.cost, feed_dict={self.input_x: input_x, self.action: action, self.reward: reward})
    #end

    ## Return the current cost
    def current_error(self, input_x, action, reward):
        return self.sess.run(self.error, feed_dict={self.input_x: input_x, self.action: action, self.reward: reward})
    #end

    def output(self, input_x):
        return self.sess.run(self.pred, feed_dict={self.input_x: input_x})
    #end

    ## Save the model variables into file
    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)
    #end

    ## Import the model variables
    def import_model(self, file_path):
        self.saver.restore(self.sess, file_path)
    #end

    ## Deep copy the DQN
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