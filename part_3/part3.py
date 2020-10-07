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
    r = solar_system.semi_major_axes * const.AU
    T = solar_system.star_temperature * np.sqrt(solar_system.star_radius * 1e3 / (2 * r))
    return (T >= T_min) * (T <= T_max)

"""
Returns required solar panel size for operating a lander at a given planet.

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

class SpaceMission:
    def __init__(self, rocket_motor=None, log_dir="logs/numerical_long.npy"):
        log_dir += ".npy" if log_dir[-4:] != ".npy" else ''
        infile = np.load(log_dir, allow_pickle=True)
        self.t = infile['times']
        self.p = infile['planet_positions'].T
        if rocket_motor == None:
            self.rocket_motor = Rocket_Chamber(username="ivero")
            self.rocket_motor.run_chamber_mp()
        else:
            self.rocket_motor = rocket_motor


    def launch(self, launch_position, launch_time):
        dt = self.t[1]
        self.r = np.zeros((len(self.t), 2))

if __name__ == "__main__":
    username = "ivero"
    seed = utils.get_seed(username)
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
