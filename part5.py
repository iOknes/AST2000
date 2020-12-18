#Egen kode
import numpy as np
import matplotlib.pyplot as plt

import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

#from rocket_chamber import Rocket_Chamber
from part3 import get_required_proximity

"""
Get planet's position between descreet points by linear interpolation.
Implemented to fix issues with smaller time steps when simulating satellite.

Arguments:
positions [time step, planet, dimension]: Array of positions for all planets
times [time step]: Array of times corresponding to "positions" first dimension
planet (int): Index of planet to calculate for (can be subscriptable)
time (float): the specific time you want the planets position at (in the same unit as "times")
fast (bool, optional): If True, will skip all complicated calculations and use nearest point in positions

Returns:
Numpy array of position of planet at specified time [dimensions] 
"""
def get_specific_planet_positions(positions, times, time, fast=False):
    if time > np.max(times) or time < np.min(times):
        raise ValueError("Time for requested position not a valid time.")
    dt = times[1] - times[0]
    previous_index = int(time / dt) #int casting always rounds down
    if time in times:
        return positions[np.where(times == time)[0][-1], :, :]
    if fast:
        time_diff = np.abs(times - time)
        i = np.where(time_diff == np.min(time_diff))[0][-1]
        #[0][-1] last of indexes in case the are two (array nesten in tuple)
        return positions[i, :, :]
    d_step = (time - times[previous_index]) / dt
    r_step = (positions[previous_index + 1, :, :] - positions[previous_index, :, :]) * d_step / dt
    return positions[previous_index, :, :] + r_step

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
def simulate_trajectory(t0, r0, v0, T, dt=None, log_dir="logs/numerical_long.npy", username=67085):
    #Load simulated planet positions and time from part 2
    log_dir += ".npy" if log_dir[-4:] != ".npy" else ''
    infile = np.load(log_dir, allow_pickle=True)
    t = infile["times"]
    r = infile["planet_positions"].T

    if type(username) is str:
        SolSys = SolarSystem(utils.get_seed(username))
    elif type(username) is int:
        SolSys = SolarSystem(username)
    else:
        raise ValueError("Username must be either string or int!")

    #Raise exception if trying to simulate further than simulated in part 2
    if t0 + T > t[-1]:
        raise ValueError("Trying to simulate trajectory for further than planetary simulations run.")
    
    #Get constants for part 2 simulation
    dt_sim = t[1] - t[0]
    if dt is None:
        dt = dt_sim

    N = int((T - t0) / dt)
    
    #Initialize arrays
    r_sat = np.zeros((N,2))
    v_sat = np.zeros((N,2))
    r_sat[0] = r0
    v_sat[0] = v0
    #test = get_required_proximity(r0, SolSys.masses, [0,0], SolSys.star_mass, 1)

    for i in range(N-1):
        r_ = get_specific_planet_positions(r, t, i * dt + t0, False)
        req_prox = get_required_proximity(r_, SolSys.masses, [0,0], SolSys.star_mass, k=10)
        dom_force = np.linalg.norm(r_, axis=1) <= req_prox
        if np.sum(dom_force) == 0:
            r_force = np.array([0,0])
            m_force = SolSys.star_mass
        else:
            r_force = r_[dom_force]
            m_force = SolSys.masses[dom_force]
        r_ = r_[dom_force]
        r_norm = np.linalg.norm(r_force - r_sat[i])
        v_sat[i+1] = v_sat[i] + (r_force - r_sat[i]) / r_norm * const.G_sol * m_force / r_norm**2 * dt
        r_sat[i+1] = r_sat[i] + v_sat[i+1] * dt

    return r_sat, v_sat

if __name__ == "__main__":
    infile = np.load("logs/numerical_long.npy", allow_pickle=True)
    t = infile["times"]
    r = infile["planet_positions"].T
    t0 = 0
    T = t0 + 0.05
    r0 = r[0,0,:] + 1e-2
    v0 = (r[1,0,:] - r[0,0,:]) / (t[1] - t[0]) * 1.5
    dt_sim = t[1] - t[0]
    dt = dt_sim * 10
    r_sat, v_sat = simulate_trajectory(t0, r0, v0, T, dt)
    for i in range(len(r[0])):
        plt.plot(r[:,i,0], r[:,i,1], label=f"Planet {i}")
    plt.plot(r_sat[:,0], r_sat[:,1], "--", label="satellite")
    plt.legend(loc="upper left")
    plt.xlabel("x position [AU]")
    plt.ylabel("y position [AU]")
    plt.axis("equal")
    plt.show()
