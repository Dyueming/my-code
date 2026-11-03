import numpy as np
import matplotlib.pyplot as plt

# Build x from -2 to 4
x = np.linspace(-2, 4, 100)
# Compute y
y = np.exp(-x)

# Create the figure
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='$y = e^{-x}$', color='blue')
plt.title('Graph of $y = e^{-x}$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.ylim(-0.5, 1.5)
plt.xlim(-2, 4)

# Show the figure
plt.show()