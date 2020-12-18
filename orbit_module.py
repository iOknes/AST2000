#Egen kode
from numba import jit
import numpy as np

@jit(cache = True, nopython = True)
def calc_orbit_EC(pos, vel, G, N, dt, sun_mass, planet_masses):
    # Euler Cromer
    v_last = vel
    M = -1 * G * (planet_masses + sun_mass)
    for i in range(1, N):
        r_mag = np.sqrt((pos[0,:,i-1]**2)+(pos[1,:,i-1]**2))
        u_r = pos[:,:,i-1] / r_mag

        a = (M / (r_mag**2)) * u_r
        v = v_last + (a * dt)

        pos[:,:,i] = pos[:,:,i-1] + v*dt
        v_last = v

    return pos

@jit(cache = True, nopython = True)
def calc_orbit_KD(pos, vel, G, N, dt, sun_mass, planet_masses):
    # Kick drift leapfrog
    v_full = vel
    M = -1 * G * (planet_masses + sun_mass)
    for i in range(1, N):
        r_mag_half = np.sqrt((pos[0,:,i-1]**2)+(pos[1,:,i-1]**2))
        u_r_half = pos[:,:,i-1] / r_mag_half

        a_half = (M / (r_mag_half**2)) * u_r_half
        v_half = v_full + (a_half * (dt/2))

        pos[:,:,i] = pos[:,:,i-1] + v_half*dt

        r_mag_full = np.sqrt((pos[0,:,i]**2)+(pos[1,:,i]**2))
        u_r_full = pos[:,:,i] / r_mag_full

        a_full = (M / (r_mag_full**2)) * u_r_full
        v_full = v_half + (a_full * (dt/2))

    return pos

@jit(cache = True, nopython = True)
def calc_orbit_HS(pos, vel, G, N, dt, sun_mass, planet_masses):
    # Half step leapfrog
    M = -1 * G * (planet_masses + sun_mass)

    r_mag_half = np.sqrt((pos[0,:,0]**2)+(pos[1,:,0]**2))
    u_r_half = pos[:,:,0] / r_mag_half
    a_half = (M / (r_mag_half**2)) * u_r_half
    v_half = vel + (a_half*dt)
    for i in range(1, N):
        pos[:,:,i] = pos[:,:,i-1] + (v_half*dt)

        r_mag = np.sqrt((pos[0,:,i-1]**2)+(pos[1,:,i-1]**2))
        u_r = pos[:,:,i-1] / r_mag
        a = (M / (r_mag**2)) * u_r

        v_half = v_half + (a*dt)

    return pos

@jit(cache = True, nopython = True)
def calc_solar_orbit_KD(pos_p, vel_p, pos_sun, vel_sun, G, N, dt,
                        sun_mass, planet_masses):
    M_inv = 1 / (np.sum(planet_masses) + sun_mass)
    sun_b_M = sun_mass * M_inv
    plan_b_M = planet_masses * M_inv

    num_plan = len(pos_p[0,:,0])
    pos_sun_temp = np.zeros((2))
    v_p_f = vel_p#np.zeros((2,num_plan))
    v_s_f = np.zeros((2,num_plan))
    v_p_h = np.zeros((2,num_plan))
    v_s_h = np.zeros((2,num_plan))
    cm_init = find_CM(pos_p[:,:,0], pos_sun[:,0], sun_mass, planet_masses)
    for i in range(num_plan):
        pos_p[:,i,0] = pos_p[:,i,0] - cm_init
    pos_sun[:,0] = pos_sun[:,0] - cm_init

    for i in range(1,N):
        cm = find_CM(pos_p[:,:,i-1], pos_sun[:,i-1], sun_mass, planet_masses)
        pos_sun[:,i-1] = pos_sun[:,i-1] - cm
        pos_sun_temp[0], pos_sun_temp[1] = 0,0
        for j in range(num_plan):
            pos_p[:,j,i-1] = pos_p[:,j,i-1] - cm
            r_vec_j_h = pos_p[:,j,i-1] - pos_sun[:,i-1]

            r_mag_h = np.sqrt(r_vec_j_h[0]**2 + r_vec_j_h[1]**2)
            u_r_h = r_vec_j_h / r_mag_h

            a_p_h = ((-G*sun_mass)/(r_mag_h**2)) * u_r_h
            a_s_h = ((-G*planet_masses[j])/(r_mag_h**2)) * u_r_h

            v_p_h[:,j] = v_p_f[:,j] + (a_p_h*(dt/2))
            v_s_h[:,j] = v_s_f[:,j] + (a_s_h*(dt/2))

            pos_p[:,j,i] = pos_p[:,j,i-1] + v_p_h[:,j]*dt
            pos_sun_temp = pos_sun_temp + v_s_h[:,j] * dt

            r_vec_j_f = pos_p[:,j,i] - pos_sun[:,i-1]

            r_mag_f = np.sqrt(r_vec_j_f[0]**2 + r_vec_j_f[1]**2)
            u_r_f = r_vec_j_f / r_mag_f

            a_p_f = ((-G*sun_mass)/(r_mag_f**2)) * u_r_f
            a_s_f = ((-G*planet_masses[j])/(r_mag_f**2)) * u_r_f

            v_p_f[:,j] = v_p_h[:,j] + (a_p_f*(dt/2))
            v_s_f[:,j] = v_s_h[:,j] + (a_s_f*(dt/2))


        pos_sun[:,i] = pos_sun[:,i-1] + pos_sun_temp
    return pos_p, pos_sun


@jit(cache = True, nopython = True)
def find_CM(pos_p, pos_s, sun_mass, planet_masses):
    cm_p = np.asarray([np.sum(pos_p[0,:] * planet_masses),
                       np.sum(pos_p[1,:] * planet_masses)])
    cm = (pos_s*sun_mass) + cm_p
    return cm

@jit(cache = True, nopython = True)
def calc_solar_orbit_KD_EN(pos_p, vel_p, pos_sun, vel_sun, G, N, dt,
                           sun_mass, planet_masses):
    M_inv = 1 / (np.sum(planet_masses) + sun_mass)
    sun_b_M = sun_mass * M_inv
    plan_b_M = planet_masses * M_inv

    num_plan = len(pos_p[0,:,0])
    pos_sun_temp = np.zeros((2))
    v_p_f = vel_p#np.zeros((2,num_plan))
    v_s_f = np.zeros((2,num_plan),dtype=np.float64)
    v_p_h = np.zeros((2,num_plan),dtype=np.float64)
    v_s_h = np.zeros((2,num_plan),dtype=np.float64)
    tot_energies = np.zeros((2, num_plan, N-1),dtype=np.float64)
    cm_init = find_CM(pos_p[:,:,0], pos_sun[:,0], sun_mass, planet_masses)
    for i in range(num_plan):
        pos_p[:,i,0] = pos_p[:,i,0] - cm_init
    pos_sun[:,0] = pos_sun[:,0] - cm_init

    for i in range(1,N):
        cm = find_CM(pos_p[:,:,i-1], pos_sun[:,i-1], sun_mass, planet_masses)
        pos_sun[:,i-1] = pos_sun[:,i-1] - cm
        pos_sun_temp[0], pos_sun_temp[1] = 0,0
        for j in range(num_plan):
            pos_p[:,j,i-1] = pos_p[:,j,i-1] - cm
            r_vec_j_h = pos_p[:,j,i-1] - pos_sun[:,i-1]

            r_mag_h = np.sqrt(r_vec_j_h[0]**2 + r_vec_j_h[1]**2)
            u_r_h = r_vec_j_h / r_mag_h

            a_p_h = ((-G*sun_mass)/(r_mag_h**2)) * u_r_h
            a_s_h = ((-G*planet_masses[j])/(r_mag_h**2)) * u_r_h

            v_p_h[:,j] = v_p_f[:,j] + (a_p_h*(dt/2))
            v_s_h[:,j] = v_s_f[:,j] + (a_s_h*(dt/2))

            pos_p[:,j,i] = pos_p[:,j,i-1] + v_p_h[:,j]*dt
            pos_sun_temp = pos_sun_temp + v_s_h[:,j] * dt

            r_vec_j_f = pos_p[:,j,i] - pos_sun[:,i-1]

            r_mag_f = np.sqrt(r_vec_j_f[0]**2 + r_vec_j_f[1]**2)
            u_r_f = r_vec_j_f / r_mag_f

            a_p_f = ((-G*sun_mass)/(r_mag_f**2)) * u_r_f
            a_s_f = ((-G*planet_masses[j])/(r_mag_f**2)) * u_r_f

            v_p_f[:,j] = v_p_h[:,j] + (a_p_f*(dt/2))
            v_s_f[:,j] = v_s_h[:,j] + (a_s_f*(dt/2))

            v_mag_p = np.sqrt(v_p_f[0,j]**2 + v_p_f[1,j]**2)
            v_mag_s = np.sqrt(v_s_f[0,j]**2 + v_s_f[1,j]**2)
            r_mag_p = np.sqrt(pos_p[:,j,i][0]**2 + pos_p[:,j,i][1]**2)
            r_mag_s = (np.sqrt((pos_sun[:,i-1] + pos_sun_temp)[0]**2 +
                               (pos_sun[:,i-1] + pos_sun_temp)[1]**2))

            tot_energies[0,j,i-1] = get_tot_energy(r_mag_s, v_mag_s,
                                                   G, planet_masses[j],
                                                   sun_mass)
            tot_energies[1,j,i-1] = get_tot_energy(r_mag_p, v_mag_p,
                                                   G, planet_masses[j],
                                                   sun_mass)

        pos_sun[:,i] = pos_sun[:,i-1] + pos_sun_temp
    return pos_p, pos_sun, tot_energies

@jit(cache = True, nopython = True)
def get_tot_energy(r_mag, v_mag, G, P_m, S_m):
    r_m = (P_m * S_m) / (P_m + S_m)
    tot_m = P_m + S_m
    return ((0.5 * r_m * (v_mag**2)) - (G*tot_m*r_m)/(r_mag))


@jit(cache=True,nopython=True)
def check_tot_energy(tot_energy, tol):

    energy_conserved = True
    msg = "Energy not sufficiently conserved"

    max = np.amax(tot_energy)
    min = np.amin(tot_energy)
    val = np.abs(max) - np.abs(min)

    if val < tol:
        energy_conserved = False

    assert energy_conserved, msg

if __name__ == "__main__":
    print("These are functions to be used in the orbit_class")
