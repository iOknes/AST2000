import numpy as np

class Sim:
    def __init__(self, dimensions=3):
        self.dim = dimensions
        self.bodyNum = 0
        self.r_init = np.zeros((0, dimensions), dtype='float64')
        self.v_init = np.zeros((0, dimensions), dtype='float64')
        self.mass = np.zeros(0, dtype='float64')
        self.forces = []

    def add_molecule(self, pos, vel, mass):
        self.bodyNum += 1
        self.r_init = np.concatenate((self.r_init, np.array([pos])))
        self.v_init = np.concatenate((self.v_init, np.array([vel])))
        self.mass = np.concatenate((self.mass, np.array([mass])))

    def add_molecules(self, pos, vel, mass):
        self.bodyNum += len(pos)
        self.r_init = np.concatenate((self.r_init, np.array(pos)))
        self.v_init = np.concatenate((self.v_init, np.array(vel)))
        self.mass = np.concatenate((self.mass, np.array(mass)))

    def add_force(self, F):
        self.forces.append(F)

    def force(self, r, mass):
        force = np.zeros_like(r)
        for f in self.forces:
            force += f(r, mass)
        return force
    def solve(self, dt, T):
        N = int(T / dt)
        
        self.r = np.zeros((N, *np.shape(self.r_init)))
        self.v = np.zeros((N, *np.shape(self.v_init)))

        self.r[0] = self.r_init
        self.v[0] = self.v_init

        """        
        for i in range(N-1):
            self.v[i+1] = self.v[i] + self.force(self.r[i], self.mass) * dt
            self.r[i+1] = self.r[i] + self.v[i+1] * dt
        """

        for i in range(N-1):
            vh = self.v[i] + self.force(self.r[i], self.mass) * dt / 2
            self.r[i+1] = self.r[i] + vh * dt
            self.v[i+1] = vh + self.force(self.r[i+1], self.mass) * dt / 2
