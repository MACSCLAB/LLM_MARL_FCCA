import copy
import numpy as np

import torch
import torch.nn as nn
import time

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def transform(robot_states, human_states, human_num, att_agents):
    """
    1. Convert the absolute coordinates of the human into coordinates relative to the robot
    2. delete the robot absolute coordinates
    robot_state:[batch, robot_obs_dim]   
    human_state:[batch, human_num, human_obs_dim]
    robot_obs:[px, py, gx, gy, v]
    human_obs:[px, py, vx, vy, theta]
    """
    if human_num < att_agents:
        human_states = human_states[:,:human_num,:]

    robot_states = robot_states.unsqueeze(1)
    human_states[:, :, 0] -= robot_states[:, :, 0]
    human_states[:, :, 1] -= robot_states[:, :, 1]
    # calculate atan2
    human_states[:, :, 4] = torch.atan2(human_states[:, :, 1], human_states[:, :, 0])

    robot_states = robot_states.squeeze(1)
    robot_states = robot_states[:,2:]
    return robot_states, human_states #new_state:[batch, human_num, human_obs_dim]


