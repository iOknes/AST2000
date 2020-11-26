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

        self.dtr = np.pi / 180 #degrees to radians

        if not path.exists(self.img_dir):
            makedirs(self.img_dir)
        if not path.exists(self.log_dir):
            makedirs(self.log_dir)

        self.username = username

        self.set_seed()
        self.SS = SolarSystem(self.seed)
        self.SM = SpaceMission(self.seed)


    def set_seed(self):
        self.seed = utils.get_seed(self.username)
        np.random.seed(self.seed)

    def kappa(self, theta_0, theta, phi_0, phi):
        num = 2
        theta_0 = theta_0 * self.dtr
        theta = theta * self.dtr
        phi = phi * self.dtr
        den = (1 + (np.cos(theta_0)*np.cos(theta)) +
              (np.sin(theta_0) * np.sin(theta) * np.cos(phi - phi_0)) )
        return (num/den)

    def rho(self, X, Y):
        return np.sqrt((X**2) + (Y**2))

    def beta(self, X, Y):
        return (2*np.arctan(self.rho(X, Y) / 2))

    def get_theta(self, theta_0, X, Y):
        theta_0 = theta_0 * self.dtr
        var1 = np.cos(self.beta(X, Y)) * np.cos(theta_0)
        var2 = (Y / self.rho(X, Y)) * np.sin(self.beta(X, Y)) * np.sin(theta_0)
        theta = (theta_0 - np.arcsin(var1 + var2))
        return theta

    def get_phi(self, phi_0, X, Y):
        phi_0 = phi_0 * self.dtr
        var1 = X * np.sin(self.beta(X, Y))
        var2 = self.rho(X, Y) * np.sin(phi_0) * np.cos(self.beta(X, Y))
        phi = phi_0 + np.arctan(var1 / var2)
        return phi

    def get_X(self, theta_0, theta, phi_0, phi):
        theta_0 = theta_0 * self.dtr
        theta = theta * self.dtr
        phi_0 = phi_0 * self.dtr
        phi = phi * self.dtr
        return (self.kappa(theta_0, theta, phi_0, phi) *
                np.sin(theta) * np.sin(phi - phi_0))

    def get_Y(self, theta_0, theta, phi_0, phi):
        theta_0 = theta_0 * self.dtr
        theta = theta * self.dtr
        phi_0 = phi_0 * self.dtr
        phi = phi * self.dtr
        var1 = np.sin(theta_0) * np.cos(theta)
        var2 = np.cos(theta_0) * np.sin(theta) * np.cos(phi - phi_0)
        return (self.kappa(theta_0, theta, phi_0, phi) * (var1 - var2))

    def get_X_max_min(self, phi_max, phi_min, max = True):
        alpha_phi = phi_max - phi_min
        alpha_phi = alpha_phi * self.dtr
        max_val = (2 * np.sin(alpha_phi/2)) / (1 + np.cos(alpha_phi/2))
        if max == False:
            min = max_val * -1
            return min
        return max_val

    def get_Y_max_min(self, theta_max, theta_min, max = True):
        alpha_theta = theta_max - theta_min
        alpha_theta = alpha_theta * self.dtr
        max_val = (2 * np.sin(alpha_theta/2)) / (1 + np.cos(alpha_theta/2))
        if max == False:
            min = max_val * -1
            return min
        return max_val

    def sample_pic(self, img_name, FOV, center):
        hk = np.load("himmelkule.npy")
        print(np.shape(hk))
        FOV_theta, FOV_phi = FOV
        theta_0, phi_0 = center
        img = Image.open(img_name)
        img_array = np.asarray(img)
        img_Y, img_X = np.shape(img_array)[0], np.shape(img_array)[1]

        phi_pix =  FOV_phi / img_X
        theta_pix = FOV_theta / img_Y

        print(phi_pix, theta_pix)

        phi_min = phi_0 - (phi_pix * (img_X / 2))
        phi_max = phi_0 + (phi_pix * (img_X / 2))

        theta_min = theta_0 - (theta_pix * (img_Y / 2))
        theta_max = theta_0 + (theta_pix * (img_Y / 2))

        print(phi_min, phi_max)
        print(theta_min, theta_max)

        X_max = self.get_X_max_min(phi_max, phi_min, max = True)
        X_min = self.get_X_max_min(phi_max, phi_min, max = False)
        Y_max = self.get_Y_max_min(theta_max, theta_min, max = True)
        Y_min = self.get_Y_max_min(theta_max, theta_min, max = False)

        print(X_max, X_min)
        print(Y_max, Y_min)

        grid_x = np.linspace(X_min, X_max, img_X)
        grid_y = np.linspace(Y_min, Y_max, img_Y)

        #grid_nx, grid_ny = np.meshgrid(grid_x, grid_y)

        grid_xy = np.zeros((img_Y, img_X, 2))
        grid_phi_theta = np.zeros((img_Y, img_X, 2))

        for i in range(img_Y):
            for j in range(img_X):
                grid_xy[i,j,0] = grid_y[i]
                grid_xy[i,j,1] = grid_x[j]

                grid_phi_theta[i,j,0] = self.get_theta(theta_0, grid_x[j],
                                                       grid_y[i])
                grid_phi_theta[i,j,1] = self.get_phi(phi_0, grid_x[j],
                                                     grid_y[i])
        #print(grid_phi_theta)
        gen_img = np.zeros((img_Y, img_X,3))

        for i in range(img_Y):
            for j in range(img_X):
                index = self.SM.get_sky_image_pixel(grid_phi_theta[i,j,0],
                                                    grid_phi_theta[i,j,1])
                gen_img[i,j,:] = hk[index,2:]
                #print(index)



        new_img = Image.fromarray(gen_img.astype(np.uint8))

        #new_img.show()
        #img.show()





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
    sample_FOV = [70, 70]
    sample_center = [90, 0]


    SC = SpaceCraft(log_name, log_dir, img_dir, username)
    SC.sample_pic(sample_img_name, sample_FOV, sample_center)
