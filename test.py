import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.array([1, 3, 5, 3, 1])
y = np.array([2, 1, 3, 1, 2])
line, = ax.plot(x, y)

ymax = max(y)
xpos = np.where(y == ymax)
xmax = x[xpos]

ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax + 5), arrowprops=dict(facecolor='black'),)

plt.show()