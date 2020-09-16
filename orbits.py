import numpy as np
import matplotlib.pyplot as plt

r = lambda f, p, e: p / (1 + e * np.cos(f))
t = np.linspace(0, 2 * np.pi, 1001)

print(np.min(r(t, 5.90638e9, 0.2488)), np.max(r(t, 5.90638e9, 0.2488)))

plt.plot(r(t, 57909050, 0.205630) * np.cos(t), r(t, 57909050, 0.205630) * np.sin(t))
plt.plot(r(t, 108208000, 0.006772) * np.cos(t), r(t, 108208000, 0.006772) * np.sin(t))
plt.plot(r(t, 149598023, 0.0167086) * np.cos(t), r(t, 149598023, 0.0167086) * np.sin(t))
plt.plot(r(t, 227939200, 0.0934) * np.cos(t), r(t, 227939200, 0.0934) * np.sin(t))
plt.plot(r(t, 778.57e6, 0.0489) * np.cos(t), r(t, 778.57e6, 0.0489) * np.sin(t))
plt.plot(r(t, 1433.53e6, 0.0565) * np.cos(t), r(t, 1433.53e6, 0.0565) * np.sin(t))
plt.plot(r(t, 2875.04e6, 0.046381) * np.cos(t), r(t, 2875.04e6, 0.046381) * np.sin(t))
plt.plot(r(t, 4.50e9, 0.008678) * np.cos(t), r(t, 4.50e9, 0.008678) * np.sin(t))
plt.plot(r(t, 5.90638e9, 0.2488) * np.cos(t), r(t, 5.90638e9, 0.2488) * np.sin(t))
plt.axis("equal")
plt.show()
