#! /usr/bin/env python

import numpy as np
from math import *
from std_msgs.msg import String
from itertools import product
from sensor_msgs.msg import LaserScan

import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

STATE_SPACE_IND_MAX = 144 - 1
STATE_SPACE_IND_MIN = 1 - 1
ACTIONS_IND_MAX = 2
ACTIONS_IND_MIN = 0

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75

T_MIN = 0.001

# Create actions
def createActions():
    actions = np.array([0,1,2])
    return actions

# Create state space for DQN
def createStateSpace():
    x1 = set((0,1,2))
    x2 = set((0,1,2))
    x3 = set((0,1,2,3))
    x4 = set((0,1,2,3))
    state_space = set(product(x1,x2,x3,x4))
    return np.array(list(state_space))

def getAction(state, epsilon, shape_action, model):
    if len(state.shape) == 1:
        state = np.expand_dims(state, axis=0)
    if np.random.rand() <= epsilon:
        return random.randrange(shape_action)
    else:
        action_vals = model.predict(state)
        return np.argmax(action_vals[0])
        
def getBestAction(state, model):
    state = np.expand_dims(state, axis=0)
    action_vals = model.predict(state)
    return np.argmax(action_vals[0])


# Reward function for Q-learning - table
def getReward(action, prev_action, lidar, prev_lidar, crash):
    if crash:
        terminal_state = True
        reward = -10
    else:
        lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
        prev_lidar_horizon = np.concatenate((prev_lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],prev_lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
        terminal_state = False
        # Reward from action taken = fowrad -> +0.2 , turn -> -0.1
        if action == 0:
            r_action = +0.2
        else:
            r_action = -0.1
        # Reward from crash distance to obstacle change
        W = np.linspace(0.9, 1.1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1.1, 0.9, len(lidar_horizon) // 2))
        if np.sum( W * ( lidar_horizon - prev_lidar_horizon) ) >= 0:
            r_obstacle = +0.2
        else:
            r_obstacle = -0.2
        # Reward from turn left/right change
        if ( prev_action == 1 and action == 2 ) or ( prev_action == 2 and action == 1 ):
            r_change = -0.8
        else:
            r_change = 0.0

        # Cumulative reward
        reward = r_action + r_obstacle + r_change

    return ( reward, terminal_state )
