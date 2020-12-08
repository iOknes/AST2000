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


def deg2rad(deg):
    return (deg / 180)*np.pi

class SpaceCraft():

    def __init__(self, log_name, log_dir, img_dir,
                 username = "YourUserName", hk = "himmelkule.npy"):

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
        self.SM = SpaceMission(self.seed)
        self.hk = np.load(hk)

    def set_seed(self):
        self.seed = utils.get_seed(self.username)
        np.random.seed(self.seed)


    def get_X_max_min(self, alpha_phi, max = True):
        max_val = (2 * np.sin(alpha_phi/2)) / (1 + np.cos(alpha_phi/2))
        if max == False:
            return (max_val * -1)
        return max_val

    def get_Y_max_min(self, alpha_theta, max = True):
        max_val = (2 * np.sin(alpha_theta/2)) / (1 + np.cos(alpha_theta/2))
        if max == False:
            return (max_val * -1)
        return max_val


    def set_mesh(self, FOV, width, height):
        alpha_theta = FOV[0]
        alpha_phi = FOV[1]

        x_max = self.get_X_max_min(alpha_phi, max = True)
        x_min = self.get_X_max_min(alpha_phi, max = False)
        y_max = self.get_Y_max_min(alpha_theta, max = True)
        y_min = self.get_Y_max_min(alpha_theta, max = False)

        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)
        return np.meshgrid(x,y)

    def XY_phi_theta(self, FOV, phi_0, theta_0, img_size):
        width, height = img_size[0], img_size[1]
        X, Y = self.set_mesh(FOV, width, height)
        rho = np.sqrt(X**2 + Y**2)
        beta = 2*np.arctan(rho/2)

        den1 = X * np.sin(beta)
        num1 = ((rho * np.sin(phi_0) * np.cos(beta))
               -(Y * np.cos(phi_0) * np.sin(beta)))
        phi = phi_0 + np.arctan(den1 / num1)

        var1 = ((np.cos(beta) * np.cos(theta_0))
               +((Y/rho) * np.sin(beta) * np.sin(theta_0)))
        theta = theta_0 - np.arcsin(var1)

        return phi, theta

    def gen_img(self, FOV, phi_0, theta_0, img_size):
        width, height = img_size[0], img_size[1]

        Phi, Theta = self.XY_phi_theta(FOV, phi_0, theta_0, img_size)

        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                theta = Theta[i,j]
                phi = Phi[i,j]
                index = self.SM.get_sky_image_pixel(theta, phi)
                pixels[i,j] = self.hk[index][2:]
        img = Image.fromarray(pixels)
        return img

    def get_sample_img_size(self, sample_img_path):
        img = Image.open(sample_img_path)
        img_array = np.asarray(img)
        img_height, img_width = np.shape(img_array)[0], np.shape(img_array)[1]
        return [img_width, img_height]






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

    #img_example_func()
    username = "ivero"
    log_name = "part4"
    log_dir = "__cache__"
    img_dir = "plots"

    sample_img_dir = "Images"
    sample_img_name = f"{sample_img_dir}/sample0000.png"
    sample_FOV = [deg2rad(70), deg2rad(70)]
    sample_phi_0 = deg2rad(0)
    sample_theta_0 = deg2rad(90)

    SC = SpaceCraft(log_name, log_dir, img_dir, username)
    img_size = SC.get_sample_img_size(sample_img_name)
    img = SC.gen_img(sample_FOV, sample_phi_0, sample_theta_0, img_size)
    img.show()
    a = np.asarray(img)
    print(np.shape(a))
