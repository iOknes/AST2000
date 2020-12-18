#Egen kode
import numpy as np
import matplotlib.pyplot as plt

import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

from rocket_chamber import Rocket_Chamber

"""
Returns an array of boolians corresponding to the planets which are within the
habitable zone of the star. The habitable zone is set to be any planet who's 
surface temperature is between 260 K and 390 K as default, but this can be set
by the user.

Arguments:
solar_system: ast2000tools.solar_system.SolarSystem instance
(optional) T_min: minimum temperature for planets to be counted as habitable (default: 260)
(optional) T_max: maximum temperature for planets to be counted as habitable (default: 390)

Returns:
habitable_planets: np.array(dtype=bool) of what planets are habitable
"""
def find_habitable_planets(solar_system, T_min=260, T_max=390):
    a = solar_system.semi_major_axes * const.AU
    T = solar_system.star_temperature * np.sqrt(solar_system.star_radius * 1e3 / (2 * a))
    return (T >= T_min) * (T <= T_max)

"""
Calculates required solar panel size for operating a lander at a given planet.

Arguments:
solar_system: ast2000tools.solar_system.SolarSystem instance where the planet and star are located
target_planet: the index of the planet in the solar system the lander will be going to
(optional) efficiency: the efficiency coeffisient of the solar panels (default: 0.12)
(optional) target_effciency: the desired effect from the solar panels in watts (default: 40)

Returns:
solar_panel_area: (float) required area for solar pnale on given planet to meet
specified effect at given effect
"""
def find_lander_panel_size(solar_system, target_planet, efficiency=0.12, target_effect=40):
    star_radius = solar_system.star_radius * 1e3
    semi_major_axis = solar_system.semi_major_axes[target_planet] * const.AU
    return target_effect * semi_major_axis**2 / \
    (efficiency * const.sigma * solar_system.star_temperature**4 * star_radius**2)

"""
Calculates best case scenario proximity to a planet for it to be considered the
dominant gravitational force on an object.

Arguments:
solar_system: ast2000tools.solar_system.SolarSystem instance where the planet and star are located
target_planet: the index of the planet in the solar system the lander will be going to

Returns:
required_proximity_distance: the best case scenario distance from a planet that
makes it the dominant force on an object
"""
def get_required_proximity_solar_system(solar_system, target_planet):
    m_s = solar_system.star_mass
    m_p = solar_system.masses[target_planet]
    r = solar_system.semi_major_axes[target_planet]
    return r * np.sqrt(m_p / (5 * m_s))

def get_required_proximity(sattelite_position, planet_masses, star_positon, star_mass, target_planet=None, k=5):
    sattelite_position = np.array(sattelite_position)
    star_positon = np.array(star_positon)
    m_s = star_mass
    if target_planet is None:
        m_p = planet_masses
    else:
        m_p = planet_masses[target_planet]
    r = np.linalg.norm(sattelite_position - star_positon)
    return r * np.sqrt(m_p / (k * m_s))

class SpaceMission:
    def __init__(self, rocket_motor=None, log_dir="logs/numerical_long.npy"):
        log_dir += ".npy" if log_dir[-4:] != ".npy" else ''
        infile = np.load(log_dir, allow_pickle=True)
        self.t = infile['times']
        self.p = infile['planet_positions'].T
        if rocket_motor == None:
            self.rocket_motor = Rocket_Chamber(username="67085", cache=False)
            self.rocket_motor.run_chamber_mp()
        else:
            self.rocket_motor = rocket_motor
        self.thrust = 5e14 * self.rocket_motor.F

    def launch(self, launch_position, launch_time):
        #Initialise paramters for simulation
        dt = self.t[1]
        N = len(self.t)
        self.r = np.zeros((len(self.t), 2))
        t_diff = np.abs(self.t - launch_time)
        i_start = np.where(t_diff == np.min(t_diff))[0][0]
        t_start = self.t[i_start]

        #Copy planets position to space ships position pre-launch.
        self.r[:i_start] = self.p[:i_start+1,0]

        for i in range(i_start, N):
            r[i] = r[i-1]

if __name__ == "__main__":
    seed = 67085
    SolSys = SolarSystem(seed)
    habitable_planets = find_habitable_planets(SolSys)
    print(np.arange(len(habitable_planets))[habitable_planets])
    print(np.array(SolSys.types)[habitable_planets])
    print(SolSys.semi_major_axes[habitable_planets] - SolSys.semi_major_axes[0])
    """From this we decide that planet 1 is our target, seeing as it is the only
    other habitable planet than our home planet"""
    target_planet = 1
    min_solar_panel_area = find_lander_panel_size(SolSys, target_planet)
    print(f"Minimum solar panel size: {min_solar_panel_area}m^2")
