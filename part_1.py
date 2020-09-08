import numpy as np
import matplotlib.pyplot as plt
from time import time
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
        np.random.seed(self.seed)
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
        self.home_planet_escape_velocity = np.sqrt(2 * const.G * self.home_planet_mass / self.home_planet_radius)

    def run_particle_box(self, dt=1e-3, pN=100000, N=1000, temp=12.373, nozzle=0.5, plot_energy=False):
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

    @staticmethod
    def fuel_consumed(thrust, fuel_consumption, initial_mass, target_speed, dt=1):
        vel = 0
        mass_wet = initial_mass
        mass_dry = 1100
        while vel < target_speed:
            mass_wet -= fuel_consumption * dt
            if mass_wet < mass_dry:
                print(f"Bummer :( We've run out of fuel at {vel:.2f}m/s!")
                break
            vel += thrust / mass_wet * dt
        return mass_wet - mass_dry
        #return fuel_consumption * initial_mass * target_speed / thrust

    def simulate_launch(self, thrust, initial_mass, speed_boost):
        fuel_consumption = self.fuel_consumption
        fuel_mass = initial_mass - 1100

if __name__ == "__main__":
    rm = RocketMotor("ivero")
    rm.run_particle_box()
    print(rm.fuel_consumed(6.67e14 * rm.f_per_box, 6.67e14 * rm.fuel_consumption, 3600, 14800))
