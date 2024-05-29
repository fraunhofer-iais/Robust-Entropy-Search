# Copyright (c) 2017 Zi Wang, changes by D. Weichert on January, 31st, 2024
from .push_world import *
import numpy as np

def robot_push_3d(rx, ry, simu_steps, gx, gy):
    # function works on unscaled parameters
    rx = float(rx)
    ry = float(ry)
    gx = float(gx)
    gy = float(gy)
    simu_steps = int(float(simu_steps)*10)
    # set it to False if no gui needed
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (
    0.3, 1)
    thing, base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0, 0))

    # if rx is zero, ret will be 0, too
    if rx == 0:
        if ry > 0:
            # we push only upwards
            init_angle = np.pi / 2
        elif ry < 0:
            # we push only downwards
            init_angle = -np.pi / 2
        else:
            # we push with random initial angle
            init_angle = np.random.rand() * np.pi - np.pi/2
    else:
        init_angle = np.arctan(ry / rx)
    robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
    ret = simu_push(world, thing, robot, base, simu_steps)
    ret = np.linalg.norm(np.array([gx, gy]) - ret)

    # for maximization
    ret = 5 - ret
    return ret

def get_goals(num_pairs):
    # first target: uniform over domain [-5, 5]
    g1 = np.random.rand(num_pairs, 2) * 10 - 5

    # second target: uniform over l1 ball centered at the first target location with radius 2
    # sample an angle theta
    theta = np.random.rand(num_pairs, 1) * np.pi * 2.
    radius = np.random.rand(num_pairs, 1) * 2.

    # get the coordinates
    g2x = g1[:, [0]] + radius * np.sin(theta)
    g2y = g1[:, [1]] + radius * np.cos(theta)

    g2 = np.concatenate([g2x, g2y], axis=1)
    return g1, g2