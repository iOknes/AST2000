import time
import matplotlib.pyplot as plt
#from numba import jit
import multiprocessing as mp
import faulthandler
import numpy as np

from os import path
from os import makedirs

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

from modules import orbit_module as o_m


class PlanetOrbits():

    def __init__(self, log_name, log_dir, img_dir,
                 username = "YourUserName",
                 n_pr = int(mp.cpu_count()-2)):

        self.username = username
        self.log_name = log_name
        self.log_dir = log_dir
        self.img_dir = img_dir

        self.G = const.G_sol

        self.set_seed()
        self.SS = SolarSystem(self.seed)
        self.get_system_data()

        if mp.cpu_count() >= 4:
            self.multiprocess = True
            if mp.cpu_count() > self.system_data["num_planets"]:
                self.n_pr = self.system_data["num_planets"]
            else:
                self.n_pr = n_pr
        else:
            self.multiprocess = False


    def set_seed(self):
        self.seed = utils.get_seed(self.username)
        np.random.seed(self.seed)

    def get_system_data(self):
        self.system_data = {}
        self.system_data["star_mass"] = self.SS.star_mass
        self.system_data["star_radius"] = self.SS.star_radius
        self.system_data["star_temperature"] = self.SS.star_temperature
        self.system_data["star_color"] = self.SS.star_color
        self.system_data["num_planets"] = self.SS.number_of_planets
        self.system_data["semi_major_axes"] = self.SS.semi_major_axes
        self.system_data["eccentricities"] = self.SS.eccentricities
        self.system_data["masses"] = self.SS.masses
        self.system_data["radii"] = self.SS.radii
        self.system_data["initial_orbital_angles"] = self.SS.initial_orbital_angles
        self.system_data["aphelion_angles"] = self.SS.aphelion_angles
        self.system_data["rotational_periods"] = self.SS.rotational_periods
        self.system_data["initial_positions"] = self.SS.initial_positions
        self.system_data["initial_velocities"] = self.SS.initial_velocities
        self.system_data["atmospheric_densities"] = self.SS.atmospheric_densities
        self.system_data["types"] = self.SS.types

    def get_analytical_pos(self, theta):
        n_p = int(self.system_data["num_planets"])
        p = (self.system_data["semi_major_axes"] *
            (1.0 - self.system_data["eccentricities"]))
        r = np.zeros((len(theta),n_p))
        for i in range(len(r)):
            r[i] = (p/(1.0 + self.system_data["eccentricities"]*
                    np.cos(theta[i]-self.system_data["initial_orbital_angles"])))
        e_x = np.zeros((len(theta),n_p))
        e_y = np.zeros((len(theta),n_p))
        ur = np.array([np.cos(theta), np.sin(theta)])
        for j in range(len(r)):
            for k in range(len(r[0])):
                e_x[j,k] = r[j,k] * ur[0,j]
                e_y[j,k] = r[j,k] * ur[1,j]

        return e_x, e_y

    def analytical_orbit(self, plot_size, filename):
        if not path.exists(self.img_dir):
            makedirs(self.img_dir)
        width, height = plot_size

        theta = np.linspace(0, (2*np.pi), int(1e5))
        e_x, e_y = self.get_analytical_pos(theta)

        n_col = int(np.ceil(self.system_data["num_planets"] / 2))

        fig = plt.figure(1, figsize=(width, height))
        ax = fig.add_subplot(1,1,1)
        ax.axis('equal')
        plt.title('Analytical Orbits')
        plt.xlabel('AU')
        plt.ylabel('AU')
        plt.plot(0,0,'k*')
        plt.axhline(0,lw=0.25,c='k')
        plt.axvline(0,lw=0.25,c='k')
        for i in range(len(e_x[0])):
            plt.plot(e_x[:,i], e_y[:,i], '--', label = "Planet %d, %s"%(i, self.system_data["types"][i]))
        #plt.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
          ncol=n_col, fancybox=True, shadow=True)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"{self.img_dir}/{filename}.png",dpi=800)

    def numerical_orbit(self, N, num_rev, filename,
                        make_plot = True, check_pos = True):

        x0 = self.system_data["initial_positions"][0] #AU
        y0 = self.system_data["initial_positions"][1] #AU
        vx0 = self.system_data["initial_velocities"][0] #AU/year
        vy0 = self.system_data["initial_velocities"][1] #AU/year

        HPOT= np.sqrt((1.0 / (self.system_data["star_mass"] +
                       self.system_data["masses"][0]))*
                      (self.system_data["semi_major_axes"][0]**3) )

        T = num_rev*HPOT
        t = np.linspace(0, T, int(N*T))
        dt = t[2]-t[1]
        N_in = int(N*T)

        self.T, self.t = T, t

        pos = np.zeros((2,len(x0),len(t)), dtype=np.float64)
        vel = np.zeros((2,len(x0)),dtype=np.float64)

        pos[0,:,0], pos[1,:,0] = x0, y0
        vel[0], vel[1] = vx0, vy0

        pos = o_m.calc_orbit_KD(pos, vel, self.G, N_in, dt,
                                self.system_data["star_mass"],
                                self.system_data["masses"])

        if make_plot == True:
            plt.figure(figsize = (9,7))
            for i in range(self.SS.number_of_planets):
                plt.plot(pos[0,i,:],pos[1,i,:], ls = "--", label = f"Planet: {i}")
                plt.plot(pos[0,i,0],pos[1,i,0], marker = "*")
                plt.plot(pos[0,i,-1],pos[1,i,-1], marker = "^")
            plt.axis("equal")
            plt.legend( )
            plt.axhline(0, lw=0.5)
            plt.axvline(0, lw=0.5)
            plt.savefig(f"{self.img_dir}/{filename}.png",dpi=800)

        if (num_rev > 20) and (check_pos == True):
            log_name = f"{self.log_dir}/{filename}.npy"
            self.SS.verify_planet_positions(simulation_duration = self.T,
                                            planet_positions = pos,
                                            filename = log_name)


    def solar_orbit_numerical(self, N, num_rev, filename,
                              make_plot = True, log_pos = True,
                              planet_ind = None):
        if planet_ind == None:
            # Reverts to heaviest planet if none specified
            masses = self.system_data["masses"]
            planet_ind = [np.where(masses ==np.max(masses))[0][0]]
            masses = masses[planet_ind]
        else:
            masses = np.zeros((len(planet_ind)))
            for i in range(len(planet_ind)):
                masses[i] = self.system_data["masses"][planet_ind[i]]

        CPOT= np.sqrt((1.0 / (self.system_data["star_mass"] +
                       self.system_data["masses"][planet_ind[0]]))*
                      (self.system_data["semi_major_axes"][planet_ind[0]]**3))

        T = num_rev*CPOT
        t = np.linspace(0, T, int(N*T))
        dt = t[2]-t[1]
        N_in = int(N*T)

        pos = np.zeros((2, len(planet_ind), N_in))
        vel = np.zeros((2, len(planet_ind)))

        pos_sun = np.zeros((2,N_in))
        vel_sun = np.zeros((2))

        for i in range(len(planet_ind)):
            pos[0,i,0] = (self.system_data["initial_positions"]
                                          [0][planet_ind[i]])
            pos[1,i,0] = (self.system_data["initial_positions"]
                                          [1][planet_ind[i]])

            vel[0,i] = (self.system_data["initial_velocities"]
                                        [0][planet_ind[i]])
            vel[1,i] = (self.system_data["initial_velocities"]
                                        [1][planet_ind[i]])

        pos_p, pos_sun = o_m.calc_solar_orbit_KD(pos, vel, pos_sun, vel_sun,
                                                 self.G, N_in, dt,
                                                 self.system_data["star_mass"],
                                                 masses)

        if make_plot == True:
            img_name = f"{self.img_dir}/{filename}_{len(planet_ind)}planets.png"
            plt.figure(figsize=(9,7))
            for i in range(len(planet_ind)):
                plt.plot(pos_p[0,i,:], pos_p[1,i,:])
            plt.plot(pos_sun[0,:], pos_sun[1,:])
            plt.axhline(0,lw=0.25)
            plt.axvline(0,lw=0.25)
            plt.axis("equal")
            plt.savefig(img_name, dpi=800)

        if log_pos == True:
            log_name = f"{self.log_dir}/{filename}_{len(planet_ind)}planets.npy"
            positions = {}
            positions["planet_positions"] = pos_p
            positions["sun_positions"] = pos_sun
            np.save(log_name, positions)


if __name__ == "__main__":

    username = "ivero"
    log_dir = "__cache__"
    img_dir = "plots"
    id = 23132
    log_name = f"log_{username}_{id}"

    N = int(1e5)
    rev = 21
    N_solar = int(1e4)
    rev_solar = 5

    SolSys = PlanetOrbits(log_name = log_name, username = username,
                          log_dir = log_dir, img_dir = img_dir)
    #SolSys.SS.print_info()
    SolSys.analytical_orbit(plot_size=(9,7), filename = "analytical_orbit")

    SolSys.numerical_orbit(N = N, num_rev = rev, filename = "numerical",
                          make_plot = True, check_pos = True)
    # Ran with heaviest planet
    SolSys.solar_orbit_numerical(N = N_solar, num_rev = rev,
                             filename = "solar_numerical",
                             make_plot = True, log_pos = True,
                             planet_ind = [2])
    # Run with 2 heaviest planets + home planet
    SolSys.solar_orbit_numerical(N = N_solar, num_rev = rev,
                             filename = "solar_numerical",
                             make_plot = True, log_pos = True,
                             planet_ind = [0,2,6])
