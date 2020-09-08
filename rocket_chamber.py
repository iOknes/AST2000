import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
import multiprocessing as mp

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem

from modules import particle_box as p_box

import faulthandler; faulthandler.enable()

class Rocket_Chamber():

    def __init__(self, temp = 3e3, time_run = 1e-9, dt = 1e-12, L = 1e-6,
                 nozzle = None, num_part = 1e5, particle_mass = const.m_H2,
                 username = "YourUsername", scaled = True,
                 n_pr = int(mp.cpu_count()-2)):

        self.username = username
        self.n_pr = n_pr

        # Physical Variables
        self.T = float(temp)
        self.L = float(L)
        self.l = float(L)
        self.m = float(particle_mass)
        self.n_p = int(num_part)
        self.P = (self.n_p * self.T * const.k_B) / (self.L**3)

        if nozzle == None:
            self.nozzle = self.L / 2
        else:
            self.nozzle = nozzle

        # Simulation Variables
        self.t = float(time_run)
        self.dt = float(dt)
        self.N = int(time_run / dt)

        # Statistical Variables
        self.k = float(const.k_B)

        # Initialisation functions
        self.set_seed()

        self.scaled = scaled
        if self.scaled == True:
            self.setup_chamber_scaled()
        else:
            self.setup_chamber()

    def set_seed(self):
        self.seed = utils.get_seed(self.username)
        np.random.seed(self.seed)

    def setup_chamber_scaled(self):
        self.k_m = 1
        self.m_m = 1
        #self.L = 1 / self.l
        self.T_0 = (self.m * (self.L**2)) / (self.k * (self.t**2)) * self.k_m
        self.T_m = self.T / self.T_0
        self.sigma_m = np.sqrt((self.k_m*self.T_m) / self.m_m)
        #self.L = 1 #/ self.l

        print(self.sigma_m)
        print(self.L)
        self.pos = np.random.uniform(low = (-self.L/2), high = (self.L/2),
                                     size = (self.n_p, 3))
        self.vel = np.random.normal(loc = 0.0, scale = self.sigma_m,
                                    size = (self.n_p, 3))

    def setup_chamber(self):
        self.sigma = np.sqrt((self.k*self.T) / self.m)
        self.pos = np.random.uniform(low = (-self.L/2), high = (self.L/2),
                                     size = (self.n_p,3))
        self.vel = np.random.normal(loc = 0.0, scale = self.sigma,
                                    size = (self.n_p,3))




    def run_chamber(self):
        """
        # DocString
        """

        self.p_esc,self.esc_vel,self.v_wall = p_box.sim_box(self.pos, self.vel,
                                                            self.L, self.nozzle,
                                                            self.N, self.dt)



        self.m_esc = self.esc_vel * self.m
        self.F = self.m_esc / self.t
        self.F_wall = ((self.v_wall / 3) * self.m) / self.t
        self.P_num = self.F_wall / (self.L**2)
        self.fuel_used = (self.p_esc / self.t) * self.m

    def run_chamber_mp(self):

        pool_arguments = []
        pos = np.array_split(self.pos, self.n_pr)
        vel = np.array_split(self.vel, self.n_pr)

        for i in range(self.n_pr):
            pool_arguments.append([pos[i],vel[i],self.L, self.nozzle,
                                   self.N, self.dt])


        pool = mp.Pool(processes=self.n_pr)
        results = pool.starmap(p_box.sim_box, pool_arguments)
        pool.terminate()

        self.p_esc = 0
        self.esc_vel = 0
        self.v_wall = 0

        for val in results:
            self.p_esc += val[0]
            self.esc_vel += val[1]
            self.v_wall += val[2]

        self.m_esc = self.esc_vel * self.m
        self.F = self.m_esc / self.t
        self.F_wall = ((self.v_wall / 3) * self.m) / self.t
        self.P_num = self.F_wall / (self.L**2)
        self.fuel_used = (self.p_esc / self.t) * self.m


    def print_data(self):
        print(f"N = {self.N:.2e}")
        print(f"n_p = {self.n_p:.2e}", "\n")
        print(f"p_esc = {self.p_esc:.2e}", "\n")
        print(f"v_esc = {self.esc_vel:.2e}", "\n")
        print(f"v_wall = {self.v_wall:.2e}", "\n")
        print(f"m_esc = {self.m_esc}", "\n")
        print(f"F = {self.F:.2e}", "\n")
        print(f"P_num = {self.P_num:.2e}", "\n")
        print(f"P = {self.P:.2e}", "\n")



    def log_sim_data(self, name = None):
        if name == None:
            name = f"log_username_{self.username}"
        log = {}
        log["P"] = self.P
        log["P_num"] = self.P_num
        log["fuel_used"] = self.fuel_used
        log["F"] = self.F
        np.save(name, log)






if __name__ == "__main__":
    RC1 = Rocket_Chamber(username = "jrevense",
                         time_run = 1e-9,
                         dt=1e-12,
                         num_part = 1e5,
                         scaled = False)
    t_0 = time.time()
    RC1.run_chamber_mp()
    t_1 = time.time()
    print(t_1 - t_0, "\n")
    RC1.log_sim_data()
    #RC1.run_chamber()
    #t_2 = time.time()
    #print(t_2 - t_1)
