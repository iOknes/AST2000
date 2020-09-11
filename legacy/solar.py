from sim import Sim
import numpy as np
import matplotlib.pyplot as plt

from time import time

def gravity(pos, mass):
    a = np.zeros((len(pos), len(pos),3))
    for i in range(len(pos)):
        for j in range(i+1,len(pos)):
            dr = pos[i] - pos[j]
            r = np.linalg.norm(dr)
            a[i,j] = 6.67e-11 * mass[i] * dr / r**3
            a[j,i] = 6.67e-11 * mass[j] * dr / r**3
    return np.sum(a, axis = 0)

sun_pos = [0, 0, 0]
sun_vel = [0, 0, 0]
sun_mass = 1.99e30

mercury_pos = [2.482457691136895E+10, 4.010991965362851E+10, 8.635523682687823E+08]
mercury_vel = [-5.056616436201271E+04, 2.850041505208587E+04, 6.966862106913164E+03]
mercury_mass = 3.285e23

venus_pos = [3.043847272216039E+10, -1.032026891160324E+11, -3.215531269702129E+09]
venus_vel = [3.332530175747475E+04 , 9.837914285608647E+03, -1.788529521785596E+03]
venus_mass = 4.867e24

earth_pos = [9.511001264805460E+10, 1.134711975365665E+11, -5.141145183004439E+07]
earth_vel = [-2.330578193434303E+04, 1.901622076647900E+04, -1.627765065719267]
earth_mass = 5.972e24

mars_pos = [-2.405974222936806E+11, -4.591598323825889E+10, 4.906815267791769E+09]
mars_vel = [5.547062227496759E+03, -2.170821906824929E+04, -5.908971621054260E+02]
mars_mass = 6.39e23

r = [sun_pos, mercury_pos, venus_pos, earth_pos, mars_pos]
v = [sun_vel, mercury_vel, venus_vel, earth_vel, mars_vel]
#m = [sun_mass, mercury_mass, venus_mass, earth_mass, mars_mass]
m = [sun_mass, 0, 0, 0, 0]

s = Sim()

s.add_molecules(r, v, m)
s.add_force(gravity)
t_s = time()
s.solve(60*60, 60 * 60 * 24 * 360)
print(f'Time spent simulating: {time() - t_s}')

for i in range(5):
    plt.plot(s.r[:,i,0], s.r[:,i,1], label='Particle without initial velocity')
plt.show()
