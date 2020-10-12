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


class SpaceCraft():

    def __init__(self, log_name, log_dir, img_dir,
                 username = "YourUserName"):

        self.log_name = log_name
        self.log_dir = log_dir
        self.img_dir = img_dir

        if not path.exists(self.img_dir):
            makedirs(self.img_dir)
        if not path.exists(self.log_dir):
            makedirs(self.log_dir)

        self.username = username

        self.set_seed()
        self.SS = SolarSystem(self.seed)
        

    def set_seed(self):
        self.seed = utils.get_seed(self.username)
        np.random.seed(self.seed)
