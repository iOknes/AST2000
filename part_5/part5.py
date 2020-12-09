import numpy as np
import matplotlib.pyplot as plt

import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

#from rocket_chamber import Rocket_Chamber

"""
Simulate trajectory for the sattelite for a given length of time at a certain
time step. The satellite is assumed to only be affected by the star while 
coasting.

Arguments:
t0: start time (yr)
r0: initial position (AU)
v0: initial velocity (AU/yr)
T: total simulation time (yr)
dt: simulation time step

Returns:
t_final: end time of simulation
r_final: satellite position after simulating
v_final: satellite velocity after simulating
"""
def simulate_trajectory(t0, r0, v0, T, dt):
    if True:
        pass

if __name__ == "__main__":
    pass
