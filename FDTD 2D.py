import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Settings
dt = 0.1
dx = 0.5
dy = 0.5
steps = 300;
points_x = 100;
points_y = 100;

# Fields
hx = np.zeros((steps, points_x, points_y));
hy = np.zeros((steps, points_x, points_y));
ez = np.zeros((steps, points_x, points_y));

# Material Properties
mu = np.ones((points_x, points_y)) # Permeability
ep = np.ones((points_x, points_y)) # Permittivity
co = np.zeros((points_x, points_y)) # Conductivity

# Run Simulation
for t in tqdm(range(1, steps)):
    for x in range(0, points_x):
        for y in range(0, points_y - 1):
            hx[t, x, y] = hx[t - 1, x, y] - dt / (mu[x, y] * dy) * (ez[t - 1, x, y + 1] - ez[t - 1, x, y])
    for x in range(0, points_x - 1):
        for y in range(0, points_y):
            hy[t, x, y] = hy[t - 1, x, y] + dt / (mu[x, y] * dx) * (ez[t - 1, x + 1, y] - ez[t - 1, x, y])
    for x in range(1, points_x):
        for y in range(1, points_y):
            ez[t, x, y] = ((1 - co[x, y] * dt / ep[x, y] * 0.5) / (1 + co[x, y] * dt / ep[x, y] * 0.5)) * ez[t - 1, x, y]
            ez[t, x, y] += 1 / (1 + co[x,y] * dt / ep[x, y] * 0.5) * (dt / (ep[x, y] * dx) * (hy[t, x, y] - hy[t, x - 1, y]) - dt / (ep[x, y] * dy) * (hx[t, x, y] - hx[t, x, y - 1]))
    ez[t, 50, 50] += np.sin(t * dt)

# Plot Results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
time = 0
im1 = ax1.imshow(ez[5], animated=True, cmap='jet')
im2 = ax2.imshow(hy[5], animated=True, cmap='jet')
im3 = ax3.imshow(hx[5], animated=True, cmap='jet')
ax1.title.set_text('Z Electric Field')
ax2.title.set_text('Y Magnetic Field')
ax3.title.set_text('X Magnetic Field')
def f1(t):
    return ez[t]
def f2(t):
    return hy[t]
def f3(t):
    return hx[t]
def updatefig(*args):
    global time
    time+=1
    im1.set_array(f1(time%steps))
    im2.set_array(f2(time%steps))
    im3.set_array(f3(time%steps))
    return im1, im2, im3
ani = animation.FuncAnimation(fig, updatefig, interval=5, blit=True)
plt.show()