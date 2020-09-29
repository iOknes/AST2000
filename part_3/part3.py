import numpy as np
import matplotlib.pyplot as plt

import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission

username = "ivero"
seed = utils.get_seed(username)
SolSys = SolarSystem(seed)

if __name__ == "__main__":
    L = const.sigma * SolSys.star_temperature**4 * (SolSys.star_radius * 1e3)**2
    r = SolSys.semi_major_axes * const.AU
    E = L / r**2
    T = (E / const.sigma)**(1/4)
    print(r)
    print(T)
    print((T >= 260) * (T <= 390))
