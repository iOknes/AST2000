import numpy as np
import matplotlib.pyplot as plt

import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

"""
Returns an array of boolians corresponding to the planets which are within the
habitable zone of the star. The habitable zone is set to be any planet who's 
surface temperature is between260 K and 390 K as default, but this can be set by
the user.

Arguments:
solar_system: ast2000tools.solar_system.SolarSystem instance
T_min: minimum temperature for planets to be counted as habitable
T_max: maximum temperature for planets to be counted as habitable

Returns:
habitable_planets: np.array(dtype=bool) of what planets are habitable
"""
def find_habitable_planets(solar_system, T_min=260, T_max=390):
    L = const.sigma * solar_system.star_temperature**4 * (solar_system.star_radius * 1e3)**2
    r = solar_system.semi_major_axes * const.AU
    E = L / r**2
    T = (E / const.sigma)**(1/4)
    return (T >= T_min) * (T <= T_max)

if __name__ == "__main__":
    username = "ivero"
    seed = utils.get_seed(username)
    SolSys = SolarSystem(seed)
    habitable_planets = find_habitable_planets(SolSys)
    print(np.arange(len(habitable_planets))[habitable_planets])
    print(np.array(SolSys.types)[habitable_planets])
