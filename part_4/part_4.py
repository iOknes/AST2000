import time
import matplotlib.pyplot as plt
#from numba import jit
import multiprocessing as mp
import faulthandler
import numpy as np

from os import path
from os import makedirs
from PIL import Image


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

    def kappa(self, theta_0, theta, phi_0, phi):
        num = 2
        den = (1 + (np.cos(theta_0)*np.cos(theta)) +
              (np.sin(theta_0) * np.sin(theta) * np.cos(phi - phi_0)) )
        return (num/den)

    def rho(self, X, Y):
        return np.sqrt((X**2) + (Y**2))

    def beta(self, X, Y):
        return (2*np.arctan(self.rho(X, Y) / 2))

    def get_theta(self, theta_0, X, Y):
        var1 = np.cos(self.beta(X, Y) * np.cos(theta_0))
        var2 = (Y / self.rho(X, Y)) * np.sin(self.beta(X, Y)) * np.sin(theta_0)
        return (theta_0 - np.arcsin(var1 + var2))

    def get_phi(self, phi_0, X, Y):
        var1 = X * np.sin(self.beta(X, Y))
        var2 = self.rho(X, Y) * np.sin(theta_0) * np.cos(self.beta(X, Y))

    def get_X(self, theta_0, theta, phi_0, phi):
        return (self.kappa(theta_0, theta, phi_0, phi) *
                np.sin(theta) * np.sin(phi - phi_0))

    def get_Y(self, theta_0, theta, phi_0, phi):
        var1 = np.sin(theta_0) * np.cos(theta)
        var2 = np.cos(theta_0) * np.sin(theta) * np.cos(phi - phi_0)
        return (self.kappa(theta_0, theta, phi_0, phi) * (var1 - var2))

    def sample_pic(self, img_name):
        img = Image.open(img_name)
        print(img.size)


def img_example_func():
    img_name = "Images/sample0000.png"
    img = Image.open(img_name)
    pixels = np.array(img)
    width = len(pixels[0,:])
    redpixs = [(255, 0, 0) for i in range(width)]
    pixels[240, :] = redpixs
    img2 = Image.fromarray(pixels)
    img2.show()

if __name__ == "__main__":

    username = "ivero"
    log_name = "part4"
    log_dir = "__cache__"
    img_dir = "plots"

    sample_img_dir = "Images"
    img_name = f"{sample_img_dir}/sample0000.png"



    SC = SpaceCraft(log_name, log_dir, img_dir, username)
    SC.sample_pic(img_name)
