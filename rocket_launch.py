#Egen kode
#Litt ubrukelig, men fortsatt egen...

import sys

n_box = float(sys.argv[1])

f = n_box * 5.144605787186109e-10 # kg m / s^2
fuel_consumption = n_box * 1.1679106879999998e-13 # kg / s
m_fuel = 3500 # kg
v_end = 14839.0 # m / s

v = 0 # m / s
t = 0

while v < v_end:
    m_fuel -= fuel_consumption
    v += f / (1100 + m_fuel)
    t += 1
    if m_fuel <= 0:
        print("Ran out of fuel :(")
        break

if v >= v_end:
    print("You've made it to space!")

print(f"Boxes: {n_box:.2e}, Vel: {v:.2e}, fuel left: {m_fuel:.2e}, steps: {t}s")
