import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
import multiprocessing as mp

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem

from modules import particle_box as p_box
from modules import xyz_module as xyz

#import faulthandler; faulthandler.enable()

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
        self.m = float(particle_mass)
        self.n_p = int(num_part)

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
        self.pressure = (self.N * self.k * self.T) / (self.L**3)

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
        self.T_0 = (self.m * (self.L**2)) / (self.k * (self.t**2)) * self.k_m
        self.T_m = self.T / self.T_0
        self.sigma_m = np.sqrt((self.k_m*self.T_m) / self.m_m)

        print(self.sigma_m)
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

        self.p_esc, self.v_esc, self.v_wall = p_box.sim_box(self.pos, self.vel,
                                                            self.L, self.nozzle,
                                                            self.N, self.dt)


        print(f"N = {self.N:.2e}")
        print(f"n_p = {self.n_p:.2e}", "\n")
        print(f"p_esc = {self.p_esc:.2e}", "\n")
        print(f"v_esc = {self.v_esc:.2e}", "\n")
        print(f"v_wall = {self.v_wall:.2e}", "\n")

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

        esc_part = 0
        esc_vel = 0
        v_wall = 0

        for val in results:
            esc_part += val[0]
            esc_vel += val[1]
            v_wall += val[2]

        print(f"N = {self.N:.2e}")
        print(f"n_p = {self.n_p:.2e}", "\n")
        print(f"p_esc = {esc_part:.2e}", "\n")
        print(f"v_ esc = {esc_vel:.2e}", "\n")
        print(f"v_wall = {v_wall:.2e}", "\n")







if __name__ == "__main__":
    RC1 = Rocket_Chamber(username = "jrevense",
                         time_run = 1e-9,
                         dt=1e-12,
                         num_part = 1e5,
                         scaled = True)
    t_0 = time.time()
    RC1.run_chamber_mp()
    t_1 = time.time()
    print(t_1 - t_0, "\n \n")
    RC1.run_chamber()
    t_2 = time.time()
    print(t_2 - t_1)


    """
N = 1.00e+03
n_p = 1.00e+05

1.23e+04

-4.49e+04

-1.50e-22

1.41e+03

4.334902048110962


N = 1.00e+04
n_p = 1.00e+05

1.23e+05

-4.48e+05

-1.50e-21

1.41e+03

42.41803288459778


N = 1.00e+05
n_p = 1.00e+05

1.23e+06

-4.48e+06

-1.50e-20

1.41e+03

562.2679972648621




    """
