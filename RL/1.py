import numpy as np
import matplotlib.pyplot as plt

# 创建 x 值，从 -2 到 4
x = np.linspace(-2, 4, 100)
# 计算 y 值
y = np.exp(-x)

# 创建图形
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

# 显示图形
plt.show()