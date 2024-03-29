#! /usr/bin/env python

import rclpy
from time import time
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from rclpy.node import Node
from std_srvs.srv import Empty
from std_srvs.srv._empty import Empty_Request
import sys
DATA_PATH = '/mnt/c/Users/lenovo/test/ros2_ql/Data'
MODULES_PATH = '/mnt/c/Users/lenovo/test/ros2_ql/scripts'
sys.path.insert(0, MODULES_PATH)
from gazebo_msgs.msg._model_state import ModelState
from geometry_msgs.msg import Twist
import random
import os
import pickle

from Qlearning import *
from Lidar import *
from Control import *
from DQN import *
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Learning parameters
ALPHA = 0.5
GAMMA = 0.9

# Softmax Q-table exploration function
T_INIT = 25
T_GRAD = 0.95
T_MIN = 0.001

# Epsilon Greedy Q-table exploration function 
epsilon_INIT = 0.99
epsilon_GRAD = 0.96
epsilon_MIN = 0.05

# 1 - Softmax , 2 - epsilon greedy
EXPLORATION_FUNCTION = 1

# Initial Robot position
X_INIT = -2.0
Y_INIT = -0.5
THETA_INIT = 0.0

RANDOM_INIT_POS = False

# Log file directory
LOG_FILE_DIR = DATA_PATH + '/Log_learning'

# Q table source file
Q_SOURCE_DIR = ''

###################################################focus here##################################################################

# Continue training from saved checkpoint
LOAD_CHECKPOINT = False

# Episode parameters
MAX_EPISODES = 400
MAX_STEPS_PER_EPISODE = 500
MIN_TIME_BETWEEN_ACTIONS = 0.0
############################################################################################################################



class LearningNode(Node):
    def __init__(self):
        super().__init__('learning_node')
        
        # Timer, Reset, Publisher, Request
        self.timer_period = .5 # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.reset = self.create_client(Empty, '/reset_simulation')
        self.setPosPub = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.dummy_req = Empty_Request()
        self.reset.call_async(self.dummy_req)
        
        # Action, State space, Q table
        self.actions = createActions() # If use, change actions in DQN.py
        self.state_space = createStateSpace() # If use, change state_space in DQN.py
        self.Q_table = createQTable(len(self.state_space), len(self.actions))

        # Learning parameters
        self.T = T_INIT
        self.epsilon = epsilon_INIT
        self.alpha = ALPHA
        self.gamma = GAMMA
        
        # Episodes, steps, rewards
        self.ep_steps = 0
        self.ep_reward = 0
        self.episode = 1
        self.crash = 0
        self.reward_max_per_episode = np.array([])
        self.reward_min_per_episode = np.array([])
        self.reward_avg_per_episode = np.array([])
        self.ep_reward_arr = np.array([])
        self.steps_per_episode = np.array([])
        self.reward_per_episode = np.array([])
        
        # initial position
        self.robot_in_pos = False
        self.first_action_taken = False
        
        # init time
        self.t_0 = self.get_clock().now()
        self.t_start = self.get_clock().now()

        # init timer
        while not (self.t_start > self.t_0):
            self.t_start = self.get_clock().now()

        self.t_ep = self.t_start
        self.t_sim_start = self.t_start
        self.t_step = self.t_start

        self.T_per_episode = np.array([])
        self.epsilon_per_episode = np.array([])
        self.t_per_episode = np.array([])

###################################################focus here##################################################################

        # Insert your own parameters here
        self.shape_state = 360
        self.shape_action = 3
        self.gamma = GAMMA
        self.epsilon = epsilon_INIT
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.memory = deque([], maxlen=2500)
        self.model = self.build_model(0.001)
        self.batch_size = 32
 
    # Model defining
    def build_model(self, learning_rate):
        if LOAD_CHECKPOINT:
            model = keras.models.load_model('models/last_checkpoint')
            with open('models/last_checkpoint/metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            self.epsilon = metadata['epsilon']
            self.memory = metadata['memory']
            print('checkpoint loaded from models/last_checkpoint')
            print('epsilon:', self.epsilon)
            print('len(memory):', len(self.memory))
            # with open('models/last_checkpoint/epsilon.txt', 'r') as f:
            #     self.epsilon = float(f.read())
        else:
            model = keras.Sequential(
                [
                keras.layers.Dense(24, input_dim = 360, activation='relu'),
                keras.layers.Dense(24, activation = 'relu'),
                keras.layers.Dense(self.shape_action, activation = 'linear')
                ]
            ) 
            model.compile(loss = 'mean_squared_error',
                        optimizer = keras.optimizers.Adam(lr = learning_rate))
        return model
    
    # Model training
    def replay_experience(self):
        state_arr = []
        next_state_arr = []
        action_arr = []
        reward_arr = []
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_arr.append(state)
            next_state_arr.append(next_state)
            action_arr.append(action)
            reward_arr.append(reward)
        reward_arr = np.asarray(reward_arr)
        next_state_arr = np.asarray(next_state_arr)
        state_arr = np.asarray(state_arr)
        
        target_arr = reward_arr + self.gamma * np.max(self.model.predict(next_state_arr, verbose=0), axis=1)
        for i in range(len(minibatch)):
            _, _, reward, _, done = minibatch[i]
            if done:
                target_arr[i] = reward
        target_future = self.model.predict(state_arr, verbose=0)
        target_future[np.arange(len(state_arr)), action_arr] = target_arr
        print('state')
        print(state_arr.shape)
        print('target_future')
        print(target_arr.shape)
        print('epsilon')
        print(self.epsilon)
        self.model.fit(state_arr, target_arr, epochs=1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # store memory (done is boolean of episode end)
    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )

###########################################################################################################################


               
    # Log (not important)
    def log_init(self):
        # Date
        now_start = datetime.now()
        dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")

        # Log date to files
        text = '\r\n' + 'SIMULATION START ==> ' + dt_string_start + '\r\n\r\n'
        print(text)
        # self.log_sim_info.write(text)
        self.log_sim_params.write(text)

        # Log simulation parameters
        text = '\r\nSimulation parameters: \r\n'
        text = text + '--------------------------------------- \r\n'
        if RANDOM_INIT_POS:
            text = text + 'INITIAL POSITION = RANDOM \r\n'
        else:
            text = text + 'INITIAL POSITION = ( %.2f , %.2f , %.2f ) \r\n' % (X_INIT,Y_INIT,THETA_INIT)
        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_EPISODES = %d \r\n' % MAX_EPISODES
        text = text + 'MAX_STEPS_PER_EPISODE = %d \r\n' % MAX_STEPS_PER_EPISODE
        text = text + 'MIN_TIME_BETWEEN_ACTIONS = %.2f s \r\n' % MIN_TIME_BETWEEN_ACTIONS
        text = text + '--------------------------------------- \r\n'
        text = text + 'ALPHA = %.2f \r\n' % ALPHA
        text = text + 'GAMMA = %.2f \r\n' % GAMMA
        if EXPLORATION_FUNCTION == 1:
            text = text + 'T_INIT = %.3f \r\n' % T_INIT
            text = text + 'T_GRAD = %.3f \r\n' % T_GRAD
            text = text + 'T_MIN = %.3f \r\n' % T_MIN
        else:
            text = text + 'epsilon_INIT = %.3f \r\n' % epsilon_INIT
            text = text + 'epsilon_GRAD = %.3f \r\n' % epsilon_GRAD
            text = text + 'epsilon_MIN = %.3f \r\n' % epsilon_MIN
        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_LIDAR_DISTANCE = %.2f \r\n' % MAX_LIDAR_DISTANCE
        text = text + 'COLLISION_DISTANCE = %.2f \r\n' % COLLISION_DISTANCE
        text = text + 'ZONE_0_LENGTH = %.2f \r\n' % ZONE_0_LENGTH
        text = text + 'ZONE_1_LENGTH = %.2f \r\n' % ZONE_1_LENGTH
        text = text + '--------------------------------------- \r\n'
        text = text + 'CONST_LINEAR_SPEED_FORWARD = %.3f \r\n' % CONST_LINEAR_SPEED_FORWARD
        text = text + 'CONST_ANGULAR_SPEED_FORWARD = %.3f \r\n' % CONST_ANGULAR_SPEED_FORWARD
        text = text + 'CONST_LINEAR_SPEED_TURN = %.3f \r\n' % CONST_LINEAR_SPEED_TURN
        text = text + 'CONST_ANGULAR_SPEED_TURN = %.3f \r\n' % CONST_ANGULAR_SPEED_TURN
        self.log_sim_params.write(text)
    
    # can ignore this
    def wait_for_message(
        node,
        topic: str,
        msg_type,
        time_to_wait=-1
    ):
        """
        Wait for the next incoming message.
        :param msg_type: message type
        :param node: node to initialize the subscription on
        :param topic: topic name to wait for message
        :time_to_wait: seconds to wait before returning
        :return (True, msg) if a message was successfully received, (False, ()) if message
            could not be obtained or shutdown was triggered asynchronously on the context.
        """
        context = node.context
        wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
        wait_set.clear_entities()

        sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return (False, None)

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                return (True, msg_info[0])

        return (False, None)


    # main function
    def timer_callback(self):
            _, msgScan = self.wait_for_message('/scan', LaserScan)
            # print('______________')
            # print(msgScan)
            step_time = (self.get_clock().now() - self.t_step).nanoseconds / 1e9
            
            # Check each step time duration
            if step_time > MIN_TIME_BETWEEN_ACTIONS:
                self.t_step = self.get_clock().now()
                if step_time > 2:
                    text = '\r\nTOO BIG STEP TIME: %.2f s' % step_time
                    # print(text)
                    # self.log_sim_info.write(text+'\r\n')

                # End of Learning only when episodes more than max
                if self.episode > MAX_EPISODES:
                    # simulation time
                    sim_time = (self.get_clock().now() - self.t_sim_start).nanoseconds / 1e9
                    sim_time_h = sim_time // 3600
                    sim_time_m = ( sim_time - sim_time_h * 3600 ) // 60
                    sim_time_s = sim_time - sim_time_h * 3600 - sim_time_m * 60

                    # real time
                    now_stop = datetime.now()
                    dt_string_stop = now_stop.strftime("%d/%m/%Y %H:%M:%S")
                    real_time_delta = (now_stop - self.now_start).total_seconds()
                    real_time_h = real_time_delta // 3600
                    real_time_m = ( real_time_delta - real_time_h * 3600 ) // 60
                    real_time_s = real_time_delta - real_time_h * 3600 - real_time_m * 60

                    # Log learning session info to file
                    text = '--------------------------------------- \r\n\r\n'
                    text = text + 'MAX EPISODES REACHED(%d), LEARNING FINISHED ==> ' % MAX_EPISODES + dt_string_stop + '\r\n'
                    text = text + 'Simulation time: %d:%d:%d  h/m/s \r\n' % (sim_time_h, sim_time_m, sim_time_s)
                    text = text + 'Real time: %d:%d:%d  h/m/s \r\n' % (real_time_h, real_time_m, real_time_s)
                    print(text)
                    # self.log_sim_info.write('\r\n'+text+'\r\n')
                    self.log_sim_params.write(text+'\r\n')
                    # Log data to file
                    saveQTable(LOG_FILE_DIR+'/Qtable.csv', self.Q_table)
                    np.savetxt(LOG_FILE_DIR+'/StateSpace.csv', self.state_space, '%d')
                    np.savetxt(LOG_FILE_DIR+'/steps_per_episode.csv', self.steps_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_per_episode.csv', self.reward_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/T_per_episode.csv', self.T_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/epsilon_per_episode.csv', self.epsilon_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_min_per_episode.csv', self.reward_min_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_max_per_episode.csv', self.reward_max_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_avg_per_episode.csv', self.reward_avg_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/t_per_episode.csv', self.t_per_episode, delimiter = ' , ')

                    # Close files and shut down node
                    # self.log_sim_info.close()
                    self.log_sim_params.close()
                    raise SystemExit
                
                # Learning doesn't end
                else:
                    ep_time = (self.get_clock().now() - self.t_ep).nanoseconds / 1e9
                    # End of an Episode
                    if self.crash or self.ep_steps >= MAX_STEPS_PER_EPISODE:
                        robotStop(self.velPub)
                        if self.crash:
                            # get crash position
                            _, odomMsg = self.wait_for_message('/odom', Odometry)
                            ( x_crash , y_crash ) = getPosition(odomMsg)
                            theta_crash = degrees(getRotation(odomMsg))

                        self.t_ep = self.get_clock().now()
                        self.reward_min = np.min(self.ep_reward_arr)
                        self.reward_max = np.max(self.ep_reward_arr)
                        self.reward_avg = np.mean(self.ep_reward_arr)
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                        text = '---------------------------------------\r\n'
                        if self.crash:
                            text = text + '\r\nEpisode %d ==> CRASH {%.2f,%.2f,%.2f}    ' % (self.episode, x_crash, y_crash, theta_crash) + dt_string
                            self.reset.call_async(self.dummy_req)
                        elif self.ep_steps >= MAX_STEPS_PER_EPISODE:
                            text = text + '\r\nEpisode %d ==> MAX STEPS PER EPISODE REACHED {%d}    ' % (self.episode, MAX_STEPS_PER_EPISODE) + dt_string
                        else:
                            text = text + '\r\nEpisode %d ==> UNKNOWN TERMINAL CASE    ' % self.episode + dt_string
                        text = text + '\r\nepisode time: %.2f s (avg step: %.2f s) \r\n' % (ep_time, ep_time / self.ep_steps)
                        text = text + 'episode steps: %d \r\n' % self.ep_steps
                        text = text + 'episode reward: %.2f \r\n' % self.ep_reward
                        text = text + 'episode max | avg | min reward: %.2f | %.2f | %.2f \r\n' % (self.reward_max, self.reward_avg, self.reward_min)
                        
                        # Write text for exploration function for q table
                        if EXPLORATION_FUNCTION == 1:
                            text = text + 'T = %f \r\n' % self.T
                        else:
                            text = text + 'epsilon = %f \r\n' % self.epsilon
                        print(text)
                        # self.log_sim_info.write('\r\n'+text)

                        self.steps_per_episode = np.append(self.steps_per_episode, self.ep_steps)
                        self.reward_per_episode = np.append(self.reward_per_episode, self.ep_reward)
                        self.T_per_episode = np.append(self.T_per_episode, self.T)
                        self.epsilon_per_episode = np.append(self.epsilon_per_episode, self.epsilon)
                        self.t_per_episode = np.append(self.t_per_episode, ep_time)
                        self.reward_min_per_episode = np.append(self.reward_min_per_episode, self.reward_min)
                        self.reward_max_per_episode = np.append(self.reward_max_per_episode, self.reward_max)
                        self.reward_avg_per_episode = np.append(self.reward_avg_per_episode, self.reward_avg)
                        self.ep_reward_arr = np.array([])
                        self.ep_steps = 0
                        self.ep_reward = 0
                        self.crash = 0
                        self.robot_in_pos = False
                        self.first_action_taken = False
                        if self.T > T_MIN:
                            self.T = T_GRAD * self.T
###################################################focus here##################################################################
                        if self.epsilon > epsilon_MIN:
                            self.epsilon = epsilon_GRAD * self.epsilon
                        self.episode = self.episode + 1
                        if len(self.memory) > self.batch_size :
                            self.replay_experience()
                            if not os.path.exists('models'):
                                os.makedirs('models')
                            self.model.save('models/last_checkpoint')
                            metadata = {
                                'epsilon': self.epsilon,
                                'memory': self.memory
                            }
                            with open("models/last_checkpoint/metadata.pkl", "wb") as f:
                                pickle.dump(metadata, f)
                            print('checkpoint saved to models/last_checkpoint')
                            print('epsilon:', self.epsilon)
                            print('len(memory):', len(self.memory))
                            # with open("models/last_checkpoint/epsilon.txt", "w") as f:
                            #     f.write(str(self.epsilon))
################################################################################################################################


                    
                    # If the Episode can continue    
                    else:
                        self.ep_steps = self.ep_steps + 1
                        # Initial position
                        if not self.robot_in_pos:
                            robotStop(self.velPub)
                            self.ep_steps = self.ep_steps - 1
                            self.first_action_taken = False
                            
                            # get random position from original
                            # don't need to use since we have fixed coordinates
                            # if RANDOM_INIT_POS:
                            #     ( x_init , y_init , theta_init ) = robotSetRandomPos(self.setPosPub)
                            # else:
                            #     ( x_init , y_init , theta_init ) = robotSetPos(self.setPosPub, X_INIT, Y_INIT, THETA_INIT)
                            
                            ( x_init , y_init , theta_init ) = robotSetPos(self.setPosPub, X_INIT, Y_INIT, THETA_INIT)
                            
                            _, odomMsg = self.wait_for_message('/odom', Odometry)
                            ( x , y ) = getPosition(odomMsg)
                            theta = degrees(getRotation(odomMsg))
                            # check init pos
                            if abs(x-x_init) < 0.01 and abs(y-y_init) < 0.01 and abs(theta-theta_init) < 1:
                                self.robot_in_pos = True
                                #sleep(2)
                            else:
                                self.robot_in_pos = False
                                
                        # First action in an Episode
                        elif not self.first_action_taken:
                            ( lidar, angles ) = lidarScan(msgScan) # get lidar data
###################################################focus here##################################################################
                            # initial n 
                            n = 1
                            if n != 1:
                                lidar = get_lidar(lidar, n)
                                self.shape_state = lidar.shape[0]
                                self.model = self.build_model(0.001)
                            else:
                                lidar = get_lidar(lidar, n)

                            # ( state_ind, x1, x2 ,x3 ,x4 ) = scanDiscretization(self.state_space, lidar)
                            self.crash = checkCrash(lidar)
                            state = lidar # define state to be lidar 
                            self.action = getAction(state, self.epsilon, self.shape_action, self.model) # getAction func in DQN.py
#########################################################################################################################
                            status_rda = robotDoAction(self.velPub, self.action)

                            self.prev_lidar = lidar
                            self.prev_action = self.action
                            self.prev_state = state
                            self.first_action_taken = True

                        # Rest of the algorithm
                        else:
                            ( lidar, angles ) = lidarScan(msgScan)
###################################################focus here##################################################################
                            # Divide lidar from 360 values to 360/n values
                            # initial n 
                            n = 1
                            if n != 1:
                                lidar = get_lidar(lidar, n)
                                self.shape_state = lidar.shape[0]
                                self.model = self.build_model(0.001)
                            else:
                                lidar = get_lidar(lidar, n)

                            # ( state_ind, x1, x2 ,x3 ,x4 ) = scanDiscretization(self.state_space, lidar)
                            self.crash = checkCrash(lidar)
                            state = lidar
                            self.shape_state = state.shape[0]
                            # get reward from getReward func in DQN.py
                            ( reward, done ) = getReward(self.action, self.prev_action, lidar, self.prev_lidar, self.crash)

                            self.memory.append( (self.prev_state, self.action, 0, state, done) ) # store in memory
                            self.action = getAction(lidar, self.epsilon, self.shape_action, self.model) # getAction func in DQN.py
#########################################################################################################################

                            status_rda = robotDoAction(self.velPub, self.action)
                            self.ep_reward = self.ep_reward + reward
                            self.ep_reward_arr = np.append(self.ep_reward_arr, reward)
                            self.prev_lidar = lidar
                            self.prev_action = self.action
                            self.prev_state = state
                            self.first_action_taken = True

def main(args=None):
    rclpy.init(args=args)
    movebase_publisher = LearningNode()
    try:
        rclpy.spin(movebase_publisher)
    except SystemExit:                 # <--- process the exception 
        rclpy.logging.get_logger("End of learning").info('Done')
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    movebase_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
