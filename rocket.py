#Egen kode

import time
import matplotlib.pyplot as plt
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
        self.planet_radius = self.SS.radii[0]*1e3 # radius in m
        self.g = (self.G * self.planet_mass_kg) / (self.planet_radius**2)


        self.escape_velocity = np.sqrt((2*self.G*self.planet_mass_kg) /
                                        (self.planet_radius))

        self.rotational_period = self.SS.rotational_periods[0] # in days
        self.rotational_speed = ((self.rotational_period/(24*60*60)) *
                                  (2*np.pi*self.planet_radius)) # m/s
        self.omega = (2*np.pi) / (24*60*60)


        #print("\n")
        #print(f"m_craft: {self.rocket_mass}")
        #print(f"ev: {self.escape_velocity:.2e}")
        #print(f"g: {self.g:.2f}")
        #print(f"N_craft: {self.g*self.rocket_mass:.2e}")
        #print("\n")


    def set_seed(self):
        if type(self.username) is str:
            self.seed = utils.get_seed(self.username)
        elif type(self.username) is int:
            self.seed = self.username
        else:
            raise ValueError("Type of username must be either string or int!")
        np.random.seed(self.seed)

    def load_sim_log(self):
        self.log = np.load(f"{self.log_name}.npy", allow_pickle = True).item()
        self.log_data = []
        for key in self.log:
            self.log_data.append(key)
            #print(f"{key}: {self.log[key]}")

    def m_to_AU(self, m):
        return (m*(6.68458712e-12))

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
                    #print("Ran out of fuel")
                    init_fuel_mass *= mult
                    fuel_mass = init_fuel_mass
                    break

            #print(f"v_rem: {dv - v:.2f}, fuel_mass: {fuel_mass:.2f}")
            if v > dv:
                print("Success!")
                break
            v = 0
            t = 0



        print(f"t: {t}, v: {v:.2e}, v_rem: {dv - v:.2e}")
        print(f"init_fuel: {init_fuel_mass:.2f}, fuel_left: {fuel_mass:.2f}")
        print(f"fuel_burned: {init_fuel_mass-fuel_mass:.2f}")


    def Engine_Performance2(self, dv, N, init_fuel_mass, dt, mult_in = 0.25, accuracy = 1e-2):

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

    def Grav_Acc(self, r):
        g = (self.G * self.planet_mass_kg) / ((self.planet_radius + r)**2)
        return g

    def Sim_Rocket_Launch(self, N, fuel_mass, dt):
        # To visualise the planet
        theta = np.linspace(0,2*np.pi, 1000)
        x,y = self.planet_radius*np.cos(theta), self.planet_radius*np.sin(theta)

        thrust = self.log["F"] * N
        f_s = self.log["fuel_used"] * N
        omega = self.omega
        self.init_fuel_mass = fuel_mass

        r = np.zeros((2))
        r[0] = self.planet_radius
        v = np.zeros((2))
        v_omega = np.zeros((2))
        a = np.zeros((2))

        self.r_0 = r

        r_mag = np.sqrt(np.sum(r**2))
        v_mag = np.sqrt(np.sum(v**2))

        v_omega[0] = -omega*r[1]
        v_omega[1] = omega*r[0]
        # ang_vel v = omega*r
        # a = thrust / (self.rocket_mass + fuel_mass ) - Grav_Acc(r)

        t = 0
        r_log = []
        liftoff_check = ((thrust/(self.rocket_mass+fuel_mass))-
                          self.Grav_Acc(r_mag))
        if liftoff_check < 0:
            print("Not enough thrust")
            return

        while v_mag < self.escape_velocity:
            #r_log.append(r)
            r_mag = np.sqrt(np.sum(r**2))
            u_r = (r / r_mag)

            a = ((thrust / (self.rocket_mass + fuel_mass)) -
                  self.Grav_Acc(r_mag)) * u_r

            v = v + (a*dt)
            v_omega[0] = -omega*r[1]
            v_omega[1] = omega*r[0]
            v_tot = v + v_omega
            v_mag = np.sqrt(np.sum(v_tot**2))

            r = r + (v_tot)*dt + (0.5 * a * (dt**2))
            t += dt
            fuel_mass -= f_s * dt
            if fuel_mass < 0:
                print("Ran out of fuel")
                print(f"v: {v_mag:.2e}")
                print(f"v_rem: {self.escape_velocity - v_mag:.2e}")
                print(f"t: {t:.2e}")
                break

            if t > (60*20):
                print("Ran out of fuel")
                print(f"v: {v_mag:.2e}")
                print("Took too long")
                print(f"t: {t:.2e}")
                break
        """
        r_log = (np.asarray(r_log))
        plt.figure(1, figsize=(9,9))
        plt.plot(x,y)
        plt.plot(r_log[:,0], r_log[:,1], marker = "*")
        plt.show()
        """
        print(f"F: {thrust}")
        print(f"v: {v_mag:.2e}")
        print(f"v_rem: {self.escape_velocity-v_mag:.2e}")
        print(f"v: {v}")
        print(f"t: {t}")
        print(f"fuel_left: {fuel_mass}")
        self.r = r
        self.v = v
        self.thrust = thrust
        self.f_s = f_s
        self.launch_time = t
        self.dt = dt * 1e-1
        #print(f"r_0: {self.r_0}")
        #print(f"r: {self.r}")

    def solar_ref_frame(self):
        home_planet_r = self.SS.initial_positions[:,0]
        self.r_0_solar = home_planet_r + self.m_to_AU(self.r_0)
        self.r_solar = home_planet_r + self.m_to_AU(self.r)


    def check_parameters(self, TOL):
        self.solar_ref_frame()
        self.SM.set_launch_parameters(thrust = self.thrust,
                                      mass_loss_rate = self.f_s,
                                      initial_fuel_mass = self.init_fuel_mass,
                                      estimated_launch_duration = self.launch_time,
                                      launch_position = self.r_0_solar,
                                      time_of_launch = TOL)

        self.SM.launch_rocket(time_step = self.dt)
        self.SM.verify_launch_result(position_after_launch = self.r)
        """
        Per nå får vi ikke verifisert vår launch, vi kommer tilbake til dette
        når vi får tid
        """


if __name__ == "__main__":
    username = 67085

    RC1 = Rocket_Chamber(username = username,
                         temp = 3e3,
                         time_run = 1e-9,
                         dt=1e-12,
                         num_part = 1e5,
                         cache = True)

    RC1.run_chamber_mp()
    #RC1.print_data()

    id = RC1.id
    dir = RC1.directory
    log_name = f"{dir}/log_{username}_{id}"

    R = Rocket(log_name = log_name, username = username)
    num_box = 5e14#6.67e14#1e14
    fuel_mass = 45000 #kg
    dt = 1e-3
    dv = R.escape_velocity

    print("\n")
    #R.Engine_Performance(dv, num_box, fuel_mass, dt, mult_in = 0.01)
    R.Sim_Rocket_Launch(num_box, fuel_mass, dt)
    R.check_parameters(TOL = 0)
