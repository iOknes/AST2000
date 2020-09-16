The particle box simulations are in the file rocket_chamber.py, the function to simulate the chamber is located in particle_box.py, this is imported in rocket_chamber and run there. We are very confident in the validity of this code.

The rocket itself is simulated in rocket.py, and this code runds the rocket_chamber code if the simulation for those parameters has not been run before. We've reached escape velocity with our calculations, but we're doing something wrong because the verification of the launch from the ast2000tools package does not agree.
