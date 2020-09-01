from numba import jit
import numpy as np

import faulthandler; faulthandler.enable()

@jit(cache=True, nopython=True)
def sim_box(pos, vel, L, nozzle, steps, dt):
    """
    UPDATE LATER
    input:
    pos: array of position values of particles
    vel: array of velocity values of particles
    L: side length of chamber
    nozzle: side length of nozzle opening
    steps: how many steps to run the sim for
    dt: timestep
    m: mass of particle

    output:
    number of escaped particles
    accumulated momentum lost through nozzle


    For loop advances one step, then checks whether particles are within
    the nozzle area in the xy-plane, then checks whether the particles
    are outside the bottom of the box in the z-dimension and within
    the nozzle in the xy-plane.

    xy_esc is a boolean array for particles that are within the nozzle
    area in the xy-plane
    z_esc is a boolean array for particles that are within the nozzle
    area in the xy-plane(from xy_esc), and outside the bottom of the
    chamber in the z-dimension

    The funcion then checks which values of z_esc are True, if True it
    adds one particle to the part_esc count, and adds that particles
    momentum to the momentum_esc. The momentum is calculated using the
    velocity of the particle in the z-direction, since this is the only
    direction that we are interested in based on our nozzle location.

    The function then check all particle positions, we use two boolean
    arrays, one that checks if the particle is outside on the positive
    side (L/2), and one for the negative side -(L/2), True if particle
    outside. We then loop over the flattened arrays, checking the
    boolean value, if the particle is outside(True), we reverse the
    velocity of the particle, and correct it's position by putting it
    back inside the chamber, corrected by how far it was outside.

    The function returns the number of escaped particles, and the
    accumulated momentum "lost" through the nozzle.
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
        vel_wall += np.sum(np.abs(part_outside * vel))
        #vel_wall += np.sum(part_outside_neg * np.abs(vel))
        #vel_wall += np.sum(part_outside_pos * np.abs(vel))
        vel = vel*turn

        # Corrects the positions of the particles which have
        # landed outside the box
        pos_p = (2*part_outside_pos*end_box)-(part_outside_pos * pos)
        pos_n = (-2*part_outside_neg*end_box)-(part_outside_neg * pos)
        pos = (part_inside * pos) + pos_p + pos_n



    return part_esc, vel_esc, vel_wall
