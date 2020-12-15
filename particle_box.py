#Egen kode

from numba import jit
import numpy as np

import faulthandler; faulthandler.enable()

@jit(cache=True, nopython=True)
def sim_box(pos, vel, L, nozzle, steps, dt):
    """
    input:
    pos: array of position values of particles
    vel: array of velocity values of particles
    L: side length of chamber
    nozzle: side length of nozzle opening
    steps: how many steps to run the sim for
    dt: timestep

    output:
    number of escaped particles
    accumulated velocity lost through nozzle
    accumulated velocity that hit all walls
    """

    part_esc = 0
    vel_esc = 0
    vel_wall = 0
    end_box = (L/2)
    ones = np.ones((len(pos), 3))

    for i in range(int(steps)):
        # Calculates the next position
        pos = pos + (vel*dt)

        # finds which particles are outside the box
        part_outside_neg = np.less(pos, -end_box)
        part_outside_pos = np.greater(pos, end_box)
        part_outside = np.logical_xor(part_outside_neg, part_outside_pos)
        part_inside = np.logical_not(part_outside)

        # finds which particles are inside the nozzle area in the xy-plane
        x_outside = (np.abs(pos[:,0]) < nozzle/2)
        y_outside = (np.abs(pos[:,1]) < nozzle/2)
        xy_esc = np.logical_and(x_outside, y_outside)

        # finds which particles are outside the nozzle in the z-plane,
        # using the values from the xy-plane
        z_outside =  (pos[:,2] > end_box)
        z_esc = np.logical_and(xy_esc, z_outside)
        part_esc += np.sum(z_esc)
        vel_esc += np.sum(z_esc*vel[:,2])

        # turns the velocities of the particles which are outside the box
        turn = ones - (2*part_outside)
        vel_wall += 2*np.sum(np.abs(part_outside * vel))
        vel = vel*turn

        # Corrects the positions of the particles which have
        # landed outside the box
        pos_p = (2*part_outside_pos*end_box)-(part_outside_pos * pos)
        pos_n = (-2*part_outside_neg*end_box)-(part_outside_neg * pos)
        pos = (part_inside * pos) + pos_p + pos_n

    return part_esc, vel_esc, vel_wall
