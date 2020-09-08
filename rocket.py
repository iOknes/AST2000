import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
import multiprocessing as mp

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission


class Rocket():

    def __init__(self, log_name, particle_mass = const.m_H2,
                 username = "YourUsername"):

        self.username = username
        self.log_name = log_name

        self.m = particle_mass
        self.solar_mass = 1.98847e30 #Solar mass in kg
        self.G = const.G # m^3  kg^-1  s^-2

        self.set_seed()
        self.load_sim_log()
        self.SM = SpaceMission(self.seed)
        self.SS = SolarSystem(self.seed)
        self.rocket_mass = self.SM.spacecraft_mass
        self.planet_mass_solar = self.SS.masses[0] # in Solar Masses
        self.planet_mass_kg = self.planet_mass_solar * self.solar_mass #kg
        self.planet_radius = self.SS.radii[0]*1e3 # radius in km
        self.g = (self.G * self.planet_mass_kg) / (self.planet_radius**2)


        self.escape_velocity = np.sqrt((2*self.G*self.planet_mass_kg) /
                                        (self.planet_radius))

        print("\n")
        print(f"m_craft: {self.rocket_mass}")
        print(f"ev: {self.escape_velocity:.2e}")
        print(f"g: {self.g:.2f}")
        print(f"N_craft: {self.g*self.rocket_mass:.2e}")
        print("\n")

    def set_seed(self):
        self.seed = utils.get_seed(self.username)
        np.random.seed(self.seed)

    def load_sim_log(self):
        self.log = np.load(f"{self.log_name}.npy", allow_pickle = True).item()
        self.log_data = []
        for key in self.log:
            self.log_data.append(key)
            print(f"{key}: {self.log[key]}")

    def Rocket_Boost(self, delta_v, num_box, fuel_mass, dt):
        N_tot_mass = self.g*(self.rocket_mass + fuel_mass)
        print(f"N_tot_mass: {N_tot_mass:.2e}")

        F = self.log["F"] * num_box
        fuel_per_sec = self.log["fuel_used"] * num_box
        v = 0
        t = 0

        print(f"F_tot_s: {F:.2e}")
        print(f"F_up: {F-N_tot_mass:.2e}")
        print("")
        print(f"fuel_tot_s: {fuel_per_sec:.2e}")
        print(f"Esc_Vel: {self.escape_velocity:.2e}")
        print("")
        fuel_needed = (F*fuel_per_sec) / ((self.rocket_mass + fuel_mass)*delta_v)
        print(f"fuel_consumed: {fuel_needed:.2e} \n")

        while v < self.escape_velocity:
            fuel_mass -= fuel_per_sec * dt
            v += (F/ (self.rocket_mass + fuel_mass)) * dt
            #v -= self.g * dt
            t += 1 * dt
            if fuel_mass < 0:
                print("Ran out of fuel")
                break


        print(f"t: {t}, Fuel_left: {fuel_mass:.2f}, v: {v:.2e}, num_boxes: {num_box:.2e}")







if __name__ == "__main__":

    log_name = "log_username_jrevense"

    R = Rocket(log_name = log_name)
    num_box = 5e13
    fuel_mass = 15000 #kg
    dt = 1e-3


    R.Rocket_Boost(R.escape_velocity, num_box, fuel_mass, dt)
