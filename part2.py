#Egen kode
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

from orbit_module import calc_orbit_KD, check_tot_energy, calc_orbit_EC
from orbit_module import calc_orbit_KD, check_tot_energy
from orbit_module import calc_solar_orbit_KD, calc_solar_orbit_KD_EN

class PlanetOrbits():

    def __init__(self, log_name, log_dir, img_dir,
                 username = "YourUserName",
                 n_pr = int(mp.cpu_count()-2)):

        self.username = username
        self.log_name = log_name
        self.log_dir = log_dir
        self.img_dir = img_dir
        if not path.exists(self.img_dir):
            makedirs(self.img_dir)
        if not path.exists(self.log_dir):
            makedirs(self.log_dir)

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
            (1.0 - self.system_data["eccentricities"]**2))
        r = np.zeros((len(theta),n_p))
        for i in range(len(r)):
            r[i] = (p/(1.0 + self.system_data["eccentricities"]*
                    np.cos(theta[i]+self.system_data["initial_orbital_angles"])))
        e_x = np.zeros((len(theta),n_p))
        e_y = np.zeros((len(theta),n_p))
        ur = np.array([np.cos(theta), np.sin(theta)])
        for j in range(len(r)):
            for k in range(len(r[0])):
                e_x[j,k] = r[j,k] * ur[0,j]
                e_y[j,k] = r[j,k] * ur[1,j]

        return e_x, e_y

    def analytical_orbit(self, plot_size, filename):
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
        plt.savefig(f"{self.img_dir}/{filename}.png",dpi=300)
        plt.close()

    def numerical_orbit(self, N, num_rev, filename,
                        make_plot = True, check_pos = True,
                        method = "LP_KD"):

        methods = {}
        methods["LP_KD"] = calc_orbit_KD
        methods["EC"] = calc_orbit_EC
        int_method = methods[method]
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

        pos = calc_orbit_KD(pos, vel, self.G, N_in, dt,
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
            plt.savefig(f"{self.img_dir}/{filename}.png",dpi=300)
            plt.close()

        if (num_rev > 20) and (check_pos == True):
            log_name = f"{self.log_dir}/{filename}.npy"
            self.SS.verify_planet_positions(simulation_duration = self.T,
                                            planet_positions = pos,
                                            filename = log_name)


    def solar_orbit_numerical(self, N, num_rev, filename,
                              make_plot = True, show_plot = True,
                              log_pos = True, planet_ind = None,
                              check_energy = True, tol = 1e-3):
        if planet_ind == None:
            # Reverts to heaviest planet if none specified
            masses = self.system_data["masses"]
            planet_ind = [np.where(masses ==np.max(masses))[0][0]]
            masses = masses[planet_ind]
        else:
            masses = np.zeros((len(planet_ind)))
            for i in range(len(planet_ind)):
                masses[i] = self.system_data["masses"][planet_ind[i]]
        sun_mass = self.system_data["star_mass"]
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

        if check_energy == True:
            pos_p,pos_sun,tot_energies = calc_solar_orbit_KD_EN(pos, vel,
                                                                pos_sun,
                                                                vel_sun,
                                                                self.G,
                                                                N_in, dt,
                                                                sun_mass,
                                                                masses)
            if make_plot == True:
                plt.figure(figsize=(9,7))
                img_name = f"{self.img_dir}/{filename}_{len(planet_ind)}"
                img_name += "_energy_planets"
                for j in range(len(planet_ind)):
                    ind = int(N*.1)
                    tot_en = tot_energies[0,j,5:] + tot_energies[1,j,5:]
                    lab = f"Planet {planet_ind[j]} and sun"
                    plt.plot(t[5+1:], tot_en, lw=0.5, label=lab)
                plt.legend()
                plt.xlabel("Time, t")
                plt.ylabel("Energy, E")
                plt.savefig(f"{img_name}_solar.png", dpi=300)
                plt.close()
        else:
            pos_p, pos_sun = calc_solar_orbit_KD(pos, vel, pos_sun, vel_sun,
                                                 self.G, N_in, dt, sun_mass,
                                                 masses)

        if make_plot == True:
            img_name = f"{self.img_dir}/{filename}_{len(planet_ind)}planets"
            plt.figure(figsize=(9,7))
            plt.plot(pos_sun[0,5:], pos_sun[1,5:], label = "Star")
            plt.axhline(0,lw=0.25)
            plt.axvline(0,lw=0.25)
            plt.axis("equal")
            plt.xlabel("AU")
            plt.ylabel("AU")
            plt.legend()
            plt.savefig(f"{img_name}_solar.png", dpi=300)

            for i in range(len(planet_ind)):
                lab_planet = f"Planet {planet_ind[i]}"
                plt.plot(pos_p[0,i,5:], pos_p[1,i,5:], label = lab_planet)
            plt.legend()
            plt.savefig(f"{img_name}.png", dpi=300)
            if show_plot == True:
                plt.show()
            plt.close()

        if log_pos == True:
            log_name = f"{self.log_dir}/{filename}_{len(planet_ind)}planets.npy"
            positions = {}
            positions["planet_positions"] = pos_p
            positions["sun_positions"] = pos_sun
            positions["time"] = t
            np.save(log_name, positions)

    def generate_light_curve(self, planet_index=0, N=1001, transit_fraction=0.8):
        t = np.linspace(0,1,N)
        star_radius = self.SS.star_radius
        p_raidus = self.SS.radii[planet_index]
        star_area = star_radius**2 * np.pi
        planet_area = p_raidus**2 * np.pi
        relative_area = planet_area / star_area
        light_curve = np.ones(N)
        start_time = int((1 - transit_fraction) * N)
        light_curve[start_time:-start_time] = 1 - relative_area
        light_curve += np.random.normal(0, 0.2, (N))
        return t, light_curve

    def load_logs(self, filename, num_plan):
        log_name_pos = f"{self.log_dir}/{filename}_{num_plan}planets.npy"
        positions = np.load(f"{log_name_pos}", allow_pickle = True).item()
        self.pos_s = positions["sun_positions"]
        self.pos_p = positions["planet_positions"]
        self.t = positions["time"]

    def plot_radial_vel(self, filename, num_plan, pec_vel = 0,
                        inclination = (np.pi/2),
                        add_noise = True):
        self.load_logs(filename, num_plan)
        t = self.t[10:-10]
        dt = self.t[2] - self.t[1]
        rad_vel = np.gradient(self.pos_s[0,10:-10], dt) * (const.AU / const.yr)
        rad_vel += pec_vel
        vel_max = np.max(np.abs(rad_vel)) * np.sin(inclination)
        if add_noise == True:
            noise = np.random.normal(loc=0.0, scale=(vel_max/5.0),
                                     size = len(t))
            rad_vel += noise


        img_name = f"{self.img_dir}/{filename}_{num_plan}planets"
        plt.figure(1, figsize = (11,7))
        plt.title("Radial Velocity Curve")
        plt.scatter(t[::10], rad_vel[::10], s = 5)
        plt.xlabel("Years [y]")
        plt.ylabel("Radial Velocity [m/s]")
        plt.savefig(f"{img_name}_scatter.png", dpi=300)
        plt.close()

        plt.figure(2, figsize = (11,7))
        plt.title("Radial Velocity Curve")
        plt.plot(t[::100], rad_vel[::100])
        plt.xlabel("Years [y]")
        plt.ylabel("Radial Velocity [m/s]")
        plt.savefig(f"{img_name}_plot.png", dpi=300)
        plt.close()

    def check_keplers_laws(self, planet_index=0, filename='numerical'):
        infile = np.load(f"{self.log_dir}/{filename}.npy", allow_pickle=True)
        t = infile['times']
        r = infile['planet_positions']
        r = r[:,planet_index,:].T
        A = np.cross(r[:-1], r[1:])/2
        relative_diff = np.abs((A[0] - A[20000]) / A[0])
        print("\nNumbers for checks of Kepler's laws:")
        print(f"Difference in area (relative): {relative_diff}")
        print(f"Distance traveled at aphelion: {np.linalg.norm(r[1] - r[0])} AU")
        print(f"Distance traveled at perihelion: {np.linalg.norm(r[20001] - r[20000])} AU")

if __name__ == "__main__":

    username = 67085
    log_dir = "__cache__"
    img_dir = "plots"
    id = 23132
    log_name = f"log_{username}_{id}"

    N = int(1e5)
    rev = 21
    N_solar = int(5e4)
    rev_solar = 5

    plots = False
    save_plots = True

    SolSys = PlanetOrbits(log_name = log_name, username = username,
                          log_dir = log_dir, img_dir = img_dir)
    SolSys.SS.print_info()

    SolSys.analytical_orbit(plot_size=(9,7), filename = "analytical_orbit")

    SolSys.numerical_orbit(N = N, num_rev = rev, filename = "numerical",
                           make_plot = True, check_pos = True, method = "LP_KD")
    SolSys.numerical_orbit(N = N, num_rev = 30, filename = "numerical_long",
                           make_plot = True, check_pos = True, method = "LP_KD")
    SolSys.numerical_orbit(N = N, num_rev = rev, filename = "numerical_wrong",
                           make_plot = True, check_pos = False, method = "EC")


    # Ran with heaviest planet
    SolSys.solar_orbit_numerical(N = N_solar, num_rev = rev_solar,
                             filename = "solar_numerical_big",
                             make_plot = save_plots, show_plot = plots,
                             log_pos = True, planet_ind = [2])
    SolSys.plot_radial_vel(filename = "solar_numerical_big", num_plan = 1,
                           add_noise = True, inclination = np.pi/3.,
                           pec_vel = 1.055)
    # Run with 2 heaviest planets + home planet
    SolSys.solar_orbit_numerical(N = N_solar, num_rev = rev_solar,
                             filename = "solar_numerical_home",
                             make_plot = save_plots, show_plot = plots,
                             log_pos = True, planet_ind = [2, 0, 6, 1])
    SolSys.plot_radial_vel(filename = "solar_numerical_home", num_plan = 4,
                           add_noise = True, inclination = np.pi/3.,
                           pec_vel = 1.055)

    SolSys.numerical_orbit(N = N, num_rev = rev, filename = "numerical",
                           make_plot = True, check_pos = True)

    SolSys.check_keplers_laws()

    light_curve = SolSys.generate_light_curve(2)
    plt.figure(figsize=(9,7))
    plt.scatter(*light_curve, "x")
    plt.xlabel("Relative time of transit")
    plt.ylabel("Relative flux")
    plt.savefig(f"{img_dir}/light_curve_relative_flux.png", dpi=300)
    print(f"Figure saved to {img_dir}/light_curve_relative_flux.png")
