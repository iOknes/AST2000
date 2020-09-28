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
    L = const.sigma * SolSys.star_temperature**4 * SolSys.star_radius**2
    r = SolSys.semi_major_axes
    E = L / r**2
    print(f"Habitable zone: {d_AU:.2f}AU +/- {d_AU * 0.05:.2f}AU")
    print((SolSys.semi_major_axes <= d_AU * 10) * (SolSys.semi_major_axes >= d_AU * 0.38))
