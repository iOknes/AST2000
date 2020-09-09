import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time
from os import path, makedirs
from modules import progress
from ast2000tools import utils
from ast2000tools import constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

class RocketMotor:
    def __init__(self, seed):
        #Generate or import seed based on arguments
        if isinstance(seed, str):
            self.seed = utils.get_seed(seed)
        elif isinstance(seed, int):
            self.seed = seed
        else:
            print("Error: seed must be either UiO username (str) or seed generated using ast2000tools (int)!")

        #Seed the random number generator with the seed given as argument    
        self.space_mission = SpaceMission(self.seed)

        #Set characteristic units
        self.l_0 = 1e-6
        self.t_0 = 1e-9
        self.v_0 = self.l_0 / self.t_0
        self.m_0 = const.m_H2
        self.e_0 = self.m_0 * self.v_0**2
        self.T_0 = self.e_0 / const.k_B

        #Set up planetary values for launch
        self.solar_system = SolarSystem(self.seed)
        self.home_planet_mass = self.solar_system.masses[0] * const.m_sun
        self.home_planet_radius = self.solar_system.radii[0] * 1e3
        self.escape_velocity = np.sqrt(2 * const.G * self.home_planet_mass / self.home_planet_radius)

    def get_id(self, dt, pN, N, temp, nozzle):
        self.id = self.seed * dt * self.t_0 * pN * N * temp * self.T_0 * nozzle * self.l_0 / self.m_0
        self.id = ''.join((str(self.id).split('.')))

    def save_cache(self):
        if not path.exists("__cache__/"):
            makedirs("__cache__/")
        cache_file = {"v_esc": self.v_esc, "n_esc": self.n_esc,
        "f_per_box": self.f_per_box, "p_per_box": self.p_per_box,
        "fuel_consumption": self.fuel_consumption}
        np.save(f"__cache__/{self.id}.npy", cache_file)

    def load_cache(self):
        if path.exists(f"__cache__/{self.id}.npy"):
            print(f"Loading cache from file. ID: {self.id}")
            cache_file = np.load(f"__cache__/{self.id}.npy", allow_pickle=True).item()
            self.v_esc = cache_file["v_esc"]
            self.n_esc = cache_file["n_esc"]
            self.f_per_box = cache_file["f_per_box"]
            self.p_per_box = cache_file["p_per_box"]
            self.fuel_consumption = cache_file["fuel_consumption"]
            return True
        else:
            print("No cache found for given parameters.")
            return False

    def run_particle_box(self, dt=1e-3, pN=100000, N=1000, temp=12.373,
    nozzle=0.5, plot_energy=False, track=False, cache=True):
        self.get_id(dt, pN, N, temp, nozzle)
        if cache:
            self.cache_loaded = self.load_cache()
        if not self.cache_loaded or not cache:
            np.random.seed(self.seed)
            pN = int(pN)
            dim = 3
            n_esc = 0
            v_esc = 0

            #Simulation paramteres
            r = np.zeros((N, pN, dim), dtype="float64")
            v = np.zeros((N, pN, dim), dtype="float64")

            r[0] = np.random.uniform(0, 1, (pN, dim))
            v[0] = np.random.normal(0, np.sqrt(temp), (pN, dim))
            
            t_start = time()
            bar = progress.Bar(N)

            for i in range(N-1):
                #Euler-Cromer integrator (moving particles)
                v[i+1] = v[i]
                r[i+1] = r[i] + v[i+1] * dt

                #Counting particles that would have escaped
                x_true = (r[i+1, :, 0] < (1 / 2 + nozzle / 2)) * (r[i+1, :, 0] > (1 / 2 - nozzle / 2))
                y_true = (r[i+1, :, 1] < (1 / 2 + nozzle / 2)) * (r[i+1, :, 1] > (1 / 2 - nozzle / 2))
                z_true = r[i+1, :, 2] < 0
                n_esc += np.sum(x_true * y_true * z_true)
                v_esc += np.sum(v[i+1, x_true * y_true * z_true, 2])

                #Collison checking and correcting
                v[i+1, r[i+1] < 0] = -v[i+1, r[i+1] < 0]
                v[i+1, r[i+1] > 1] = -v[i+1, r[i+1] > 1]
                r[i+1, r[i+1] < 0] = -r[i+1, r[i+1] < 0]
                r[i+1, r[i+1] > 1] = r[i+1, r[i+1] > 1] - 2 * (r[i+1, r[i+1] > 1] - 1)
                bar()
            
            if plot_energy:
                e_k = np.sum(v**2 / 2, axis=(1,2))
                plt.plot(e_k)
                plt.savefig("kinetic_energy.png")

            print(f"Calculation time: {time() - t_start}s")
            
            self.v_esc = v_esc * self.v_0
            self.n_esc = n_esc

            self.f_per_box = np.abs(v_esc * self.m_0 * self.v_0 / self.t_0)
            self.p_per_box = np.abs(v_esc * self.m_0 * self.v_0)
            self.fuel_consumption = n_esc * self.m_0 / self.t_0

        if cache:
            self.save_cache()

    @staticmethod
    def fuel_consumed(thrust, fuel_consumption, initial_mass, target_speed,
    mass_dry=1100, dt=1):
        t = 0
        vel = 0
        mass_wet = initial_mass
        mass_dry = 1100
        while vel < target_speed:
            mass_wet -= fuel_consumption * dt
            if mass_wet < mass_dry:
                print(f"Bummer :( We've run out of fuel at {vel:.2f}m/s!")
                break
            vel += thrust / mass_wet * dt
            t += 1
        print(f"Time spent accelerating: {t * dt}s")
        return initial_mass - mass_wet

    def simulate_launch(self, thrust, initial_mass, speed_boost):
        fuel_consumption = self.fuel_consumption
        fuel_mass = initial_mass - mass_dry

if __name__ == "__main__":
    n_box = 5e14
    rm = RocketMotor("ivero")
    rm.run_particle_box()
    dfuel = 12.5
    fuel = 14000
    consumed_fuel = fuel + 1
    while consumed_fuel >= fuel:
        fuel += dfuel
        consumed_fuel = rm.fuel_consumed(n_box * rm.f_per_box, n_box * rm.fuel_consumption,
        1100 + fuel, rm.escape_velocity, dt=1e-3)
    print(consumed_fuel)