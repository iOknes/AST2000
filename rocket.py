import time
#import matplotlib.pyplot as plt
#from numba import jit
import multiprocessing as mp
import faulthandler
import numpy as np

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

from rocket_chamber import Rocket_Chamber


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
        #print(f"m_craft: {self.rocket_mass}")
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
            #print(f"{key}: {self.log[key]}")

    def Engine_Performance(self, dv, N, init_fuel_mass, dt, mult_in = 0.25):
        thrust = self.log["F"] * N
        f_s = self.log["fuel_used"] * N
        fuel_mass = init_fuel_mass
        init_mass = self.rocket_mass + fuel_mass

        mult = 1 + mult_in

        v = 0
        t = 0

        while True:
            while v < dv:
                v += (thrust / (self.rocket_mass + fuel_mass)) * dt
                fuel_mass -= f_s * dt
                t += dt
                if fuel_mass < 0:
                    print("Ran out of fuel")
                    init_fuel_mass *= mult
                    fuel_mass = init_fuel_mass
                    break

            print(f"v_rem: {dv - v:.2f}, fuel_mass: {fuel_mass:.2f}")
            if v > dv:
                print("Success!")
                break
            v = 0
            t = 0



        print(f"t: {t}, v: {v:.2e}, v_rem: {dv - v:.2e}")
        print(f"init_fuel: {init_fuel_mass:.2f}, fuel_left: {fuel_mass:.2f}")
        print(f"fuel_burned: {init_fuel_mass-fuel_mass:.2f}")


    def Engine_Performance2(self, dv, N, init_fuel_mass, dt, mult_in = 0.25,
                            accuracy = 1e-2):

        thrust = self.log["F"] * N
        f_s = self.log["fuel_used"] * N
        fuel_mass = init_fuel_mass
        init_mass = self.rocket_mass + fuel_mass

        mult = 1 + mult_in

        v = 0
        t = 0
        i = 0

        while True:
            while v < dv:
                v += (thrust / (self.rocket_mass + fuel_mass)) * dt
                fuel_mass -= f_s * dt
                t += dt
                if fuel_mass < 0:
                    print("Ran out of fuel")
                    init_fuel_mass *= mult
                    fuel_mass = init_fuel_mass
                    break

            print(f"v_rem: {dv - v:.2f}, fuel_mass: {fuel_mass:.2f}")

            if np.abs(dv - v ) < accuracy:
                print("success")
                break

            #if v > dv:
            init_fuel_mass /= mult
            fuel_mass = init_fuel_mass
            mult_in /= 2
            #print(mult_in)
            mult = 1 + mult_in

            i+=1
            if i > 20:
                break


            v = 0
            t = 0



        print(f"t: {t}, v: {v:.2e}, v_rem: {dv - v:.2e}")
        print(f"init_fuel: {init_fuel_mass:.2f}, fuel_left: {fuel_mass:.2f}")
        print(f"fuel_burned: {init_fuel_mass-fuel_mass:.2f}")


    def Sim_Rocket_Launch(self, N, fuel_mass, dt):
        thrust = self.log["F"] * N
        f_s = self.log["fuel_used"] * N







if __name__ == "__main__":

    username = "jrevense"

    RC1 = Rocket_Chamber(username = username,
                         temp = 3e3,
                         time_run = 1e-9,
                         dt=1e-13,
                         num_part = 1e5)
    t_0 = time.time()
    RC1.run_chamber_mp()
    t_1 = time.time()
    print(t_1 - t_0, "\n")
    #RC1.print_data()

    id = RC1.id
    dir = RC1.directory
    log_name = f"{dir}/log_{username}_{id}"

    R = Rocket(log_name = log_name)
    num_box = 6.67e14#1e14
    fuel_mass = 2500 #kg
    dt = 1e-3
    dv = 1000 #R.escape_velocity

    #R.Rocket_Boost(R.escape_velocity, num_box, fuel_mass, dt)
    #R.Engine_Performance2(dv, num_box, fuel_mass, dt, mult_in = 0.1)
    print("\n")
    R.Engine_Performance(dv, num_box, fuel_mass, dt, mult_in = 0.1)
