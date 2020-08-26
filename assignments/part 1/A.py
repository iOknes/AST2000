import numpy as np

#The normal probability distribution function
f = lambda m, s, x: np.exp(- ((x - m) / s)**2 / 2) / (np.sqrt(2 * np.pi) * s)

def gauss(a, b, mu=0, sigma=1):
    """
    Returns probability between a and b for normal distribution of mu and sigma.

    Numerically integrates the normal distribution function with mean in mu and
    standard deviation of sigma between a and b. It is assumed that a > b.

    Parameters:
    a (float): the lower boundry of the probability interval in sigmas
    b (float): the upper boundry of the probability interval in sigmas
    mu (float): the normal distributions mean
    sigma (float): the normal distributions standard devation

    Returns:
    float: probability variable being between a and b for the given distribution
    """
    x = np.linspace(a, b, 1001)
    return np.trapz(f(mu, sigma, x), x)

#P(a <= x <= b) means the probability of x having a value between a & b.
print(f"1 stddiv out: {gauss(-1, 1)}")
print(f"2 stddiv out: {gauss(-2, 2)}")
print(f"3 stddiv out: {gauss(-3, 3)}")
