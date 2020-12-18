#Egen kode

import time
import multiprocessing as mp
import faulthandler
import numpy as np

#import os.path
from os import path
from os import makedirs

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem

from particle_box import sim_box

faulthandler.enable()

class Rocket_Chamber():

    def __init__(self, temp = 3e3, time_run = 1e-9, dt = 1e-12, L = 1e-6,
                 nozzle = None, num_part = 1e5, particle_mass = const.m_H2,
                 username = "YourUsername", directory = "__cache__",
                 n_pr = int(mp.cpu_count()-2), cache = True):

        self.username = username
        self.directory = directory
        self.cache = cache
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
        self.setup_chamber()
        self.get_id()

        self.log_name = f"{self.directory}/log_{self.username}_{self.id}"


    def set_seed(self):
        if type(seed) is str:
            self.seed = utils.get_seed(self.username)
        elif type(seed) is int:
            self.seed = self.username
        else:
            raise ValueError(f"Username argument must be either string or int, not {type(seed)}!")
        np.random.seed(self.seed)

    def get_id(self):
        id = (self.seed * self.T * self.L * self.m * self.n_p * self.P *
              self.t * self.dt * self.nozzle * self.sigma)
        id = f"{id:.32e}"
        id = "".join(id.split("."))
        self.id = id.split("e")[0]
        #print(self.id)

    def setup_chamber(self):
        self.set_seed()
        self.sigma = np.sqrt((self.k*self.T) / self.m)
        self.pos = np.random.uniform(low = (-self.L/2), high = (self.L/2),
                                     size = (self.n_p,3))
        self.vel = np.random.normal(loc = 0.0, scale = self.sigma,
                                    size = (self.n_p,3))

    def run_chamber(self, print_data = False):
        """
        # DocString
        """
        if self.cache == True:
            if path.exists(f"{self.log_name}.npy"):
                print("This simulation has already been run")
                return

        self.p_esc,self.esc_vel,self.v_wall = sim_box(self.pos, self.vel,
                                                      self.L, self.nozzle,
                                                      self.N, self.dt)

        self.m_esc = self.esc_vel * self.m
        self.F = self.m_esc / self.t
        self.F_wall = ((self.v_wall / 3) * self.m) / self.t
        self.P_num = self.F_wall / (self.L**2)
        self.fuel_used = (self.p_esc / self.t) * self.m
        self.log_sim_data()
        if print_data == True:
            self.print_data()

    def run_chamber_mp(self, print_data = False):
        if self.cache == True:
            if path.exists(f"{self.log_name}.npy"):
                print("This simulation has already been run")
                return

        pool_arguments = []
        pos = np.array_split(self.pos, self.n_pr)
        vel = np.array_split(self.vel, self.n_pr)

        for i in range(self.n_pr):
            pool_arguments.append([pos[i],vel[i],self.L, self.nozzle,
                                   self.N, self.dt])

        pool = mp.Pool(processes=self.n_pr)
        results = pool.starmap(sim_box, pool_arguments)
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
        self.F_wall = ((self.v_wall / 6) * self.m) / self.t
        self.P_num = self.F_wall / (self.L**2)
        self.fuel_used = (self.p_esc / self.t) * self.m
        self.log_sim_data()
        if print_data == True:
            self.print_data()

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

    def log_sim_data(self):
        if not path.exists(self.directory):
            makedirs(self.directory)

        log = {}
        log["P"] = self.P
        log["P_num"] = self.P_num
        log["fuel_used"] = self.fuel_used
        log["F"] = self.F
        np.save(self.log_name, log)






if __name__ == "__main__":
    RC1 = Rocket_Chamber(username = 67085,
                         temp = 3e3,
                         time_run = 1e-9,
                         dt=1e-12,
                         num_part = 1e5,
                         cache = True)
    t_0 = time.time()
    RC1.run_chamber_mp(print_data = True)
    t_1 = time.time()
    print(t_1 - t_0, "\n")
