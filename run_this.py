#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2
import multiprocessing
import tensorflow as tf
import threading
import sys
import time
import os
import deepmind_lab
import pandas as pd
import shutil


def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass


OUTPUT_GRAPH = True
LOG_DIR = './log'
lab = False
load_model = False
train = True
test_display = False
test_write_video = False
path_work_dir = "~/rl_3d/"
show_graph = True

model_path = path_work_dir + "model_lab_a3c/"

MAP = 'seekavoid_arena_01'
# MAP = 'stairway_to_melon'
N_A = 11

device = "/cpu:0"

# Hyper parameters
learning_rate = 0.00025  # 0.00025
gamma = 0.99
entropy_beta = 0.01

num_workers = multiprocessing.cpu_count()
t_max = 30
frame_repeat = 1  # 4

step_num = int(2.5e5)  # int(2.5e5)
save_each = 0.01 * step_num
step_load = 100

grad_norm_clip = 40.0

global_scope_name = "global"
step = 0
train_scores = []
loss_buf = []
lock = threading.Lock()
start_time = 0

# Global.
env = None

channels = 3
resolution = (40, 40, channels)  # (40,40,channels)

MakeDir(model_path)
model_name = model_path + "a3c"


def map_action(action):
    ACTIONS = [
    np.array([-20, 0, 0, 0, 0, 0, 0], dtype=np.intc),  # 'look_left'
    np.array([20, 0, 0, 0, 0, 0, 0], dtype=np.intc),  # 'look_right'
    np.array([0, 10, 0, 0, 0, 0, 0], dtype=np.intc),  # 'look_up'
    np.array([0, -10, 0, 0, 0, 0, 0], dtype=np.intc),  # 'look_down'
    np.array([0, 0, -1, 0, 0, 0, 0], dtype=np.intc),  # 'strafe_left'
    np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.intc),  # 'strafe_right'
    np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc),  # 'forward'
    np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.intc),  # 'backward'
    np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.intc),  # 'fire'
    np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.intc),  # 'jump'
    np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.intc)  # 'crouch'
    ]
    return ACTIONS[action]

def PrintStat(elapsed_time, step, step_num, train_scores):
    steps_per_s = 1.0 * step / elapsed_time
    steps_per_m = 60.0 * step / elapsed_time
    steps_per_h = 3600.0 * step / elapsed_time
    steps_remain = step_num - step
    remain_h = int(steps_remain / steps_per_h)
    remain_m = int((steps_remain - remain_h * steps_per_h) / steps_per_m)
    remain_s = int((steps_remain - remain_h * steps_per_h - remain_m * steps_per_m) / steps_per_s)
    elapsed_h = int(elapsed_time / 3600)
    elapsed_m = int((elapsed_time - elapsed_h * 3600) / 60)
    elapsed_s = int((elapsed_time - elapsed_h * 3600 - elapsed_m * 60))
    print("{}% | Steps: {}/{}, {:.2f}M step/h, {:02}:{:02}:{:02}/{:02}:{:02}:{:02}".format(
        100.0 * step / step_num, step, step_num, steps_per_h / 1e6,
        elapsed_h, elapsed_m, elapsed_s, remain_h, remain_m, remain_s), file=sys.stderr)

    mean_train = 0
    std_train = 0
    min_train = 0
    max_train = 0
    if (len(train_scores) > 0):
        train_scores = np.array(train_scores)
        mean_train = train_scores.mean()
        std_train = train_scores.std()
        min_train = train_scores.min()
        max_train = train_scores.max()
    print("Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
        len(train_scores), mean_train, std_train, min_train, max_train), file=sys.stderr)


def Preprocess(frame):
    if channels == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (resolution[1], resolution[0]))
    return np.reshape(frame, resolution)


class ACNet(object):
    def __init__(self, num_actions, scope, trainer_A, trainer_C):
        self.scope = scope
        with tf.variable_scope(scope):
            with tf.variable_scope("network"):
                self.inputs = tf.placeholder(shape=[None] + list(resolution), dtype=tf.float32)  # states s input

                conv1 = tf.contrib.layers.conv2d(self.inputs, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
                conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=32, kernel_size=[3, 3], stride=[2, 2])
                conv2_flat = tf.contrib.layers.flatten(conv2)
                hidden = tf.contrib.layers.fully_connected(conv2_flat, 256)

                # Recurrent network for temporal dependencies
                # Introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
                rnn_in = tf.expand_dims(hidden, [0])
                lstm_size = 256
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
                step_size = tf.shape(self.inputs)[:1]

                c_init = np.zeros((1, lstm_cell.state_size.c), dtype=np.float32)
                h_init = np.zeros((1, lstm_cell.state_size.h), dtype=np.float32)
                self.state_init = [c_init, h_init]
                self.rnn_state = self.state_init

                c_in = tf.placeholder(shape=[1, lstm_cell.state_size.c], dtype=tf.float32)
                h_in = tf.placeholder(shape=[1, lstm_cell.state_size.h], dtype=tf.float32)
                self.state_in = (c_in, h_in)

                state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in,
                                                             sequence_length=step_size, time_major=False)
                lstm_c, lstm_h = lstm_state
                rnn_out = tf.reshape(lstm_outputs, [-1, lstm_size])
                self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

            # Output layers for policy and value estimations
            with tf.variable_scope('actor'):
                self.policy = tf.contrib.layers.fully_connected(rnn_out, num_actions, activation_fn=tf.nn.softmax,
                                                            weights_initializer=self.normalized_columns_initializer(0.01),
                                                            biases_initializer=None)
            with tf.variable_scope('critic'):
                self.value = tf.contrib.layers.fully_connected(rnn_out, 1, activation_fn=None,
                                                           weights_initializer=self.normalized_columns_initializer(1.0),
                                                           biases_initializer=None)

            self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/network') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/network') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != global_scope_name:
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                policy_loss = tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = policy_loss + (entropy_beta * entropy)

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.grads_A = tf.gradients(self.policy_loss, local_vars)
                self.grads_C = tf.gradients(self.value_loss, local_vars)

                if grad_norm_clip != None:
                    grads_a, _ = tf.clip_by_global_norm(self.grads_A, grad_norm_clip)
                    grads_c, _ = tf.clip_by_global_norm(self.grads_C, grad_norm_clip)
                else:
                    grads_a = self.grads_A
                    grads_c = self.grads_C

                # Apply local gradients to global network
                self.apply_A_grads = trainer_A.apply_gradients(zip(grads_a, self.a_params))
                self.apply_C_grads = trainer_C.apply_gradients(zip(grads_c, self.c_params))


    # Used to initialize weights for policy and value output layers
    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def Train(self, sess, discounted_rewards, states, actions, advantages):
        states = states / 255.0
        self.ResetLstm()
        feed_dict = {self.target_v : discounted_rewards,
                     self.inputs : np.stack(states, axis=0),
                     self.actions : actions,
                     self.advantages : advantages,
                     self.state_in[0] : self.rnn_state[0],
                     self.state_in[1] : self.rnn_state[1]}
        _ = sess.run([self.apply_A_grads, self.apply_C_grads], feed_dict=feed_dict)

        a_l = sess.run(self.policy_loss, feed_dict=feed_dict)
        c_l = sess.run(self.value_loss, feed_dict=feed_dict)
        loss_buf.append([a_l, c_l, 0])
        data = pd.DataFrame(loss_buf, columns=['a_loss', 'c_loss', 'total_loss'])
        data.to_csv('loss_buf.csv', mode='w', index=False)

        i = sess.run(self.policy, feed_dict=feed_dict)
        j = sess.run(self.actions_onehot, feed_dict=feed_dict)
        x = sess.run(self.responsible_outputs, feed_dict=feed_dict)
        y = sess.run(self.actions, feed_dict=feed_dict)

    def ResetLstm(self):
        self.rnn_state = self.state_init

    def GetAction(self, sess, state):
        state = state / 255.0
        a_dist, v, self.rnn_state = sess.run([self.policy, self.value, self.state_out],
                                             feed_dict={self.inputs: [state],
                                                        self.state_in[0]: self.rnn_state[0],
                                                        self.state_in[1]: self.rnn_state[1]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a, v[0, 0]

    def GetValue(self, sess, state):
        state = state / 255.0
        v = sess.run([self.value],
                        feed_dict={self.inputs: [state],
                                   self.state_in[0]: self.rnn_state[0],
                                   self.state_in[1]: self.rnn_state[1]})
        return v[0][0, 0]

class Worker(object):
    def __init__(self, number, num_actions, trainer_A, trainer_C, model_name):

        self.name = "worker_" + str(number)
        self.number = number
        self.model_name = model_name
        self.pull_from_global()
        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_ac = ACNet(num_actions, self.name, trainer_A, trainer_C)

        self.env = deepmind_lab.Lab(MAP, ['RGB_INTERLEAVED', 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT'])

    # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network.
    def pull_from_global(self):
        a_globals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope_name + '/network') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope_name + '/actor')
        c_globals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope_name + '/network') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope_name + '/critic')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/network') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/network') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/critic')

        a_holder = []
        c_holder = []
        for from_var, to_var in zip(a_globals, a_params):
            a_holder.append(to_var.assign(from_var))
        for from_var, to_var in zip(c_globals, c_params):
            c_holder.append(to_var.assign(from_var))

        return a_holder, c_holder

    # Calculate discounted returns.
    def Discount(self, x, gamma):
        for idx in reversed(range(len(x) - 1)):
            x[idx] += x[idx + 1] * gamma
        return x

    def Start(self, session, saver, coord):
        worker_process = lambda: self.Process(session, saver, coord)
        thread = threading.Thread(target=worker_process)
        thread.start()

        global start_time
        start_time = time.time()
        return thread

    def Train(self, episode_buffer, sess, bootstrap_value):
        episode_buffer = np.array(episode_buffer)
        states = episode_buffer[:, 0]
        actions = episode_buffer[:, 1]
        rewards = episode_buffer[:, 2]

        values = episode_buffer[:, 3]


        # Here we take the rewards and values from the episode_buffer, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.Discount(rewards_plus, gamma)[:-1]

        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
        advantages = self.Discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        self.local_ac.Train(sess, discounted_rewards, states, actions, advantages)

    def Process(self, sess, saver, coord):
        global step, train_scores, start_time, lock, raw_data

        print("Starting worker " + str(self.number))
        while (not coord.should_stop()):
            sess.run(self.pull_from_global())
            episode_buffer = []
            episode_reward = 0

            self.env.reset()
            time_step = 0
            s = self.env.observations()['RGB_INTERLEAVED']
            s = Preprocess(s)
            self.local_ac.ResetLstm()

            while (self.env.is_running()):
                # if self.name == "worker_0":
                #     print("round : ", time_step)
                print("step : ", step)
                # Take an action using probabilities from policy network output.
                a, v = self.local_ac.GetAction(sess, s)
                act = map_action(a)
                r = self.env.step(act)
                finished = not self.env.is_running()
                if (not finished):
                    s1 = self.env.observations()['RGB_INTERLEAVED']
                    s1 = Preprocess(s1)
                else:
                    s1 = None

                episode_buffer.append([s, a, r, v])

                episode_reward += r
                s = s1

                lock.acquire()

                step += 1

                if (step % save_each == 0):
                    model_name_curr = self.model_name + "_{:04}".format(int(step / save_each))
                    print("\nSaving the network weigths to:", model_name_curr, file=sys.stderr)
                    saver.save(sess, model_name_curr)

                    PrintStat(time.time() - start_time, step, step_num, train_scores)

                    raw_data = pd.DataFrame(train_scores, columns=['train_scores'])
                    raw_data.to_csv("train_buf.csv", mode='w', index=False)
                    # train_scores = []

                if (step == step_num):
                    coord.request_stop()

                lock.release()

                # If the episode hasn't ended, but the experience buffer is full, then we
                # make an update step using that experience rollout.
                if (len(episode_buffer) == t_max or (finished and len(episode_buffer) > 0)):
                    # Since we don't know what the true final return is,
                    # we "bootstrap" from our current value estimation.
                    if not finished:
                        v1 = self.local_ac.GetValue(sess, s)
                        self.Train(episode_buffer, sess, v1)
                        episode_buffer = []
                        sess.run(self.pull_from_global())
                    else:
                        self.Train(episode_buffer, sess, 0.0)

                time_step += 1

            print("DONE!!!!!!!!!!")
            print(train_scores)
            lock.acquire()
            train_scores.append(episode_reward)
            lock.release()


class Agent(object):
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)

        with tf.device(device):
            # Global network
            self.global_net = ACNet(N_A, global_scope_name, None, None)

            if train:
                trainer_A = tf.train.RMSPropOptimizer(learning_rate, name='RMSPropA')
                trainer_C = tf.train.RMSPropOptimizer(learning_rate, name='RMSPropC')

                workers = []
                for i in range(num_workers):
                    workers.append(Worker(i, N_A, trainer_A, trainer_C, model_name))

        saver = tf.train.Saver(max_to_keep=100)
        if load_model:
            model_name_curr = model_name + "_{:04}".format(step_load)
            print("Loading model from: ", model_name_curr)
            saver.restore(self.session, model_name_curr)
        else:
            self.session.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            if os.path.exists(LOG_DIR):
                shutil.rmtree(LOG_DIR)
            tf.summary.FileWriter(LOG_DIR, self.session.graph)

        if train:
            coord = tf.train.Coordinator()
            # Start the "work" process for each worker in a separate thread.
            worker_threads = []
            for worker in workers:
                thread = worker.Start(self.session, saver, coord)
                worker_threads.append(thread)
            coord.join(worker_threads)

    def Reset(self):
        self.global_net.ResetLstm()

    def Act(self, state):
        action, _ = self.global_net.GetAction(self.session, state)
        return action


def Test(agent):
    reward_total = 0
    num_episodes = 1

    env.reset()
    agent.Reset()
    image_buffer = []
    print("running . . .")
    while (num_episodes != 0):
        if (not env.is_running()):
            env.reset()
            agent.Reset()
            print("Total reward: {}".format(reward_total))

            r = 'Y'
            if test_write_video:
                if reward_total > 3:
                    for i in range(len(image_buffer)):
                        cv2.imwrite("video/" + str(i) + ".jpg", image_buffer[i])
                        print("write")
                    print("Success!")
                    r = 'N'
                # s = input("Save or Not? (Y/N) : ")
                # if s == 'Y':
                #     for i in range(len(image_buffer)):
                #         cv2.imwrite("video/" + str(i) + ".jpg", image_buffer[i])
                #         print("write")
                #     print("Success!")

            # r = input("Play again? (Y/N) : ")

            if r == 'N':
                num_episodes -= 1
            else:
                image_buffer = []
                print("running . . .")
                reward_total = 0


        state_raw = env.observations()['RGB_INTERLEAVED']

        state = Preprocess(state_raw)
        action = agent.Act(state)
        act = map_action(action)

        if (test_display):
            cv2.imshow("frame-test", state_raw)
            cv2.waitKey(2)

        if (test_write_video):
            image_buffer.append(state_raw)

        reward = env.step(act)
        reward_total += reward


if __name__ == '__main__':
    env = deepmind_lab.Lab(MAP, ['RGB_INTERLEAVED'])
    agent = Agent()
    if not train:
        Test(agent)
