#Egen kode
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

import ast2000tools.constants as const

@jit(cache=True, nopython=True)
def MB_val(sigma, v):
    var1 = (1 / (np.sqrt(2*np.pi)*sigma))
    var2 = np.exp(((-v**2)/(2*(sigma**2))))
    return (var1*var2)

@jit(cache=True, nopython=True)
def MB_abs_val(v, T, k, m):
    var1 = ((m/(2*np.pi*k*T))**(3/2))
    var2 = np.exp(-0.5 * (m*(v**2))/(k*T))
    var3 = 4 * np.pi * (v**2)
    return (var1*var2*var3)

class MaxwellBoltzmanDist():

    def __init__(self, N = int(1e5), T = 3e3, m = const.m_H2):
        self.N = N
        self.T = T
        self.m = m
        self.k = const.k_B
        self.sigma = np.sqrt((self.k*self.T)/self.m)

    def velocity_dist(self, vx_low, vx_high, mult, name, inc_mid=True):
        vx = np.linspace(vx_low*mult, vx_high*mult, self.N)
        MB_x = MB_val(self.sigma, vx)

        ticks = np.arange(vx[0],vx[-1]+1,((abs(vx[0])+abs(vx[-1]))*0.1))

        plt.figure(1, figsize=(9,5))
        plt.title("Velocity Distribution $ v_{x} $")
        plt.plot(vx, MB_x, c="k")
        if inc_mid == True:
            plt.axvline((vx_high-vx_low), c="k", lw=0.2)

        plt.xlabel(r'$v_{x}\ in \ m/s$')
        plt.ylabel(r'$share\ of\ particles$')
        plt.xticks(ticks)

        plt.savefig('plots/%s.png'%(name))
        #plt.show()
        plt.clf()

    def integ_vel_dist(self, vx_low, vx_high, mult):
        dx = ((vx_high*mult)-(vx_low*mult))/self.N
        vx = np.linspace(vx_low*mult, vx_high*mult, self.N)
        S = MB_val(self.sigma, vx) * dx
        S_sum = np.sum(S)
        SN = S_sum * self.N
        return S_sum, SN

    def abs_velocity_dist(self, vx_low, vx_high, mult, name):
        vx = np.linspace(vx_low*mult, vx_high*mult, self.N)
        MB_x = MB_abs_val(vx, self.T, self.k, self.m)

        ticks = np.arange(vx[0],vx[-1]+1,((abs(vx[0])+abs(vx[-1]))*0.1))

        plt.figure(1, figsize=(9,5))
        plt.title('Velocity Distribution $ v $')
        plt.plot(vx, MB_x, 'k')

        plt.axvline((vx_high-vx_low), c="k", lw=0.2)
        plt.xlabel(r'$v\ in \ m/s$')
        plt.ylabel(r'$share\ of\ particles$')
        plt.xticks(ticks)
        plt.savefig("plots/%s.png"%(name))
        plt.clf()



A = MaxwellBoltzmanDist()
A.velocity_dist(-2.5, 2.5, 1e4, '2.1')
A.velocity_dist(5, 30, 1e3, '2.2', inc_mid = False)
A.abs_velocity_dist(0, 3, 1e4, '2.3')
S_sum, SN = A.integ_vel_dist(5, 30, 1e3)
print(S_sum, SN)
