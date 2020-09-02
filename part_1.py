import numpy as np
import matplotlib.pyplot as plt
from ast2000tools import utils

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
        np.random.seed(seed)

        #Set characteristic units
        self.l_0 = 1e-6
        self.t_0 = 1e-9
        self.v_0 = self.l_0 / self.t_0
        self.m_0 = 3.3476e-27
        self.e_0 = self.m_0 * self.v_0**2

    def run_particle_box(self, dt=1e-3, pN=10000, N=1000, temp=12.373, nozzle=0.5, plot_energy=False):
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
        
        self.f_per_box = v_esc
        self.fuel_consumption = n_esc
        return r, v, n_esc, v_esc

    def fuel_consumed(sattelite_mass, target_speed):
        return self.fuel_consumption * target_speed / (self.f_per_box / mass)
