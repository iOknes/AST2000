import time
import matplotlib.pyplot as plt
from numba import jit
import multiprocessing as mp
import faulthandler
import numpy as np

from os import path
from os import makedirs
from PIL import Image
from sympy.core.symbol import Symbol
from sympy.solvers.solvers import solve

import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission


from ast2000tools.shortcuts import SpaceMissionShortcuts


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
        self.H_alpha = 656.3 # nm

        #self.shortcut = SpaceMissionShortcuts(self.SM)

    def set_seed(self):
        self.seed = utils.get_seed(self.username)
        np.random.seed(self.seed)
        print(self.seed)

    def deg2rad(self, deg):
        return (deg / 180)*np.pi

    def mps_2_AUyr(self, mps):
        return ((const.yr * mps) / const.AU)

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
        num1 = ((rho * np.sin(theta_0) * np.cos(beta))
               -(Y * np.cos(theta_0) * np.sin(beta)))
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
                pixels[(height-1)-i,j] = self.hk[index][2:]
        img = Image.fromarray(pixels)
        return img


    def save_img(self, FOV, phi_0, theta_0, img_size, img_name):

        img = self.gen_img(FOV, phi_0, theta_0, img_size)
        img.save(f"{img_name}.png")


    def get_sample_img_size(self, sample_img_path):
        img = Image.open(sample_img_path)
        img_array = np.asarray(img)
        img_height, img_width = np.shape(img_array)[0], np.shape(img_array)[1]
        return [img_width, img_height]


    def make_360_pics(self, tar_dir, FOV, theta_0, img_size, img_name):
        if not path.exists(tar_dir):
            makedirs(tar_dir)

        for i in range(360):
            phi_0 = self.deg2rad(i)
            name = f"{img_name}_{i}"
            self.save_img(FOV, phi_0, theta_0, img_size, f"{tar_dir}/{name}")


    def make_360_array(self, tar_dir, img_name, img_size, array_name):
        width, height = img_size[0], img_size[1]

        ref_array = np.zeros((360, height, width, 3), dtype=np.uint8)

        for i in range(360):
            img_names = f"{tar_dir}/{img_name}_{i}.png"
            img = np.array(Image.open(img_names))
            ref_array[i] = img

        np.save(f"{array_name}.npy", ref_array)


    def determine_angle(self, img_in, array_name, img_size):
        width, height = img_size[0], img_size[1]
        sat_img = Image.open(img_in)
        img = np.array(sat_img, dtype=np.uint8)
        ref_array = np.load(f"{array_name}.npy")


        @jit(cache=True, nopython=True)
        def sum_differences(ref_array, img, height, width):
            diff = np.zeros(360)
            for i in range(360):
                diff_arr = ref_array[i] - img
                #print(i/360.*100)
                for j in range(height):
                    for k in range(width):
                        diff[i] += np.sum(diff_arr[j,k]**2)
            return diff

        diff = sum_differences(ref_array, img, height, width)

        #np.save("diff_test.npy", diff)
        min_ind = np.argmin(diff)

        if (type(min_ind) == np.ndarray) and (len(min_ind) == 2):
            theta = (min_ind[0] + min_ind[1]) * 0.5
        else:
            theta = min_ind
        return theta


    def test_determine_angle(self, img_name, array_name, img_size):
        """
        Tests the determine_angle function, it takes a while to run
        """
        width, height = img_size[0], img_size[1]

        for i in range(360):
            img_i = f"{img_name}_{i}.png"
            theta = self.determine_angle(img_i, array_name, img_size)

            msg = "Test angle not the same as determined angle"

            assert i == theta, msg

    def get_ref_stars(self):
        phi0, phi1 = self.SM.star_direction_angles
        d_lambda0, d_lambda1 = self.SM.star_doppler_shifts_at_sun
        phi = self.deg2rad(np.asarray([phi0, phi1]))
        d_lamba = np.asarray([d_lambda0, d_lambda1])
        return phi, d_lambda

    def peculiar_velocity(self, d_lambda):
        pec_v = (-d_lambda / self.H_alpha) * const.c
        return pec_v

    def vel(phi, d_lambda):
        k = (1 / np.sin(phi[1] - phi[0]))
        v_r = (-d_lambda/self.H_alpha)*const.c - self.peculiar_velocity(d_lambda)
        v_x = (k( * (np.sin(phi[1])) * v_r[0]) - (np.sin(phi[0] * v_r[1])))
        v_y = (k( * (-np.cos(phi[1])) * v_r[0])  (np.cos(phi[0] * v_r[1])))
        v = (np.asarray([v_x, v_y]) * const.yr) / const.AU
        return v

    def load_orbit_data(self, orbit_dir, orbit_file):
        # Get orbit data from part 2
        fn = f"{orbit_dir}/{orbit_file}"
        self.orbits = np.load(fn)

    def load_engine_data(self, engine_dir, engine_file):
        # Get engine data from part 1
        fn = f"{engine_dir}/{engine_file}"
        self.engine = np.load(fn)

    def xy_to_r(self, xy):
        xy_r = np.sqrt(xy[0]**2 + xy[1]**2)
        return xy_r

    def get_craft_position(self):
        #Gets craft position in relation to star, from launch from planet
        pass

    def find_vel(self, d_lambda_craft):
        phi, d_lambda_star = self.get_ref_stars()
        d_lambda_craft = self.SM.measure_star_doppler_shifts()
        v_doppler = self.vel(phi, d_lambda_craft)
        return v_doppler

    def find_distances(self):
        # Finds distances to planets from craft
        self.distances = self.SM.measure_distances()

    




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
    tar_dir = "360_imgs"

    sample_img_dir = "Images"
    sample_img_name = f"{sample_img_dir}/sample0000.png"

    SC = SpaceCraft(log_name, log_dir, img_dir, username)

    sample_FOV = [SC.deg2rad(70), SC.deg2rad(70)]
    sample_phi_0 = SC.deg2rad(0)
    sample_theta_0 = SC.deg2rad(90)
    test_img = "test_comparison_sample0000"


    img_size = SC.get_sample_img_size(sample_img_name)
    #SC.save_img(sample_FOV, sample_phi_0, sample_theta_0, img_size, test_img)

    img_360 = "himmelkule_"
    #SC.make_360_pics(tar_dir, sample_FOV, sample_theta_0, img_size, img_360)

    #SC.make_360_array(tar_dir, img_360, img_size, "himmelkule_ref")

    """
    This runs the test_determine_angle function, and takes a while to run
    It has been run with success a few times to make sure our
    determine_angle function works
    """
    #test_angle_pic_name = f"{tar_dir}/{img_360}"
    #SC.test_determine_angle(test_angle_pic_name, "himmelkule_ref", img_size)
