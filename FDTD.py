import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from FDTD import fdtd

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
fdtd(dt, dx, dy, steps, points_x, points_y, hx, hy, ez, mu, ep, co)

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