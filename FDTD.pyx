import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.double
ctypedef np.double_t DTYPE_t

def fdtd(double dt, double dx, double dy, int steps, int points_x, int points_y,
         np.ndarray [DTYPE_t, ndim=3] hx, np.ndarray [DTYPE_t, ndim=3] hy, np.ndarray [DTYPE_t, ndim=3] ez,
         np.ndarray [DTYPE_t, ndim=2] mu, np.ndarray [DTYPE_t, ndim=2] ep, np.ndarray [DTYPE_t, ndim=2] co):
    cdef int t, x, y
    for t in range(1, steps):
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