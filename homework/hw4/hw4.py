#%% imports

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

#%% q2 part d

def centrifugal_governor_system(t, state, coefs):

    alpha, beta, epsilon = coefs
    x, y, z = state

    dx = y
    dy = np.sin(x)*(z**2*np.cos(x) - 1.0) - epsilon*y
    dz = alpha*(np.cos(x) - beta)

    return [dx, dy, dz]


# Model parameters
alpha = 1.0
beta = 0.5

epsilon_crit = 2.0*alpha*beta**(3/2)

# Positive equilibrium
X_star = np.array([np.arccos(beta), 0.0, 1.0/np.sqrt(beta)])

# Small perturbation about equilibrium
state0 = X_star + np.array([0.03, 0.0, 0.0])

# Compare parameter regimes
epsilon_cases = [1.2*epsilon_crit, 0.8*epsilon_crit]
labels = ['Stable equilibrium', 'Attracting limit cycle']

# Numerical settings
t_span = (0.0, 600.0)
N = 12000
t_eval = np.linspace(*t_span, N)

fig = plt.figure(figsize=(12, 8))

for k, epsilon in enumerate(epsilon_cases):

    coefs = [alpha, beta, epsilon]

    sol = solve_ivp(
        fun=centrifugal_governor_system,
        t_span=t_span,
        y0=state0,
        t_eval=t_eval,
        args=(coefs,),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    x, y, z = sol.y

    # Remove transients for the phase portrait
    mask = sol.t >= 450.0

    # Time evolution
    ax1 = fig.add_subplot(2, 2, 2*k + 1)

    ax1.plot(sol.t, x, lw=0.8, label='x(t)')
    ax1.axhline(X_star[0], color='k', ls='--', label='Equilibrium')

    ax1.set_xlabel('t')
    ax1.set_ylabel('x(t)')
    ax1.set_title(f'{labels[k]}: epsilon = {epsilon:.4f}')
    ax1.legend()

    # 3D phase portrait
    ax2 = fig.add_subplot(2, 2, 2*k + 2, projection='3d')

    ax2.plot(x, y, z, lw=0.5, alpha=0.2, label='Transient')
    ax2.plot(x[mask], y[mask], z[mask], lw=0.8, label='Long-time trajectory')

    ax2.scatter(*X_star, color='k', s=35, label='Equilibrium')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title(labels[k])
    ax2.view_init(elev=25, azim=-60)
    ax2.legend()

    # Measure the late-time oscillation amplitude
    amplitude = 0.5*np.ptp(x[mask])
    print(f'epsilon = {epsilon:.5f}, amplitude = {amplitude:.6f}')

plt.tight_layout()
plt.show()
 


#%% q5 part a

def rossler_driver_system(t, state, coefs):

    a, b, c = coefs
    x, y, z = state

    dx = -y - z
    dy = x + a*y
    dz = b + z*(x - c)

    return [dx, dy, dz]


# Model parameters
a = 0.2
b = 0.2
c = 5.7

coefs = [a, b, c]

# Initial condition
state0 = [1.0, 1.0, 1.0]

# Numerical settings
t_span = (0.0, 300.0)
N = 20000
t_eval = np.linspace(*t_span, N)

sol = solve_ivp(
    fun=rossler_driver_system,
    t_span=t_span,
    y0=state0,
    t_eval=t_eval,
    args=(coefs,),
    rtol=1e-9,
    atol=1e-11
)

if not sol.success:
    raise RuntimeError(sol.message)

x, y, z = sol.y

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(sol.t, x, label='x(t)', lw=0.8)
ax.plot(sol.t, y, label='y(t)', lw=0.8)
ax.plot(sol.t, z, label='z(t)', lw=0.8)

ax.set_xlabel('t')
ax.set_ylabel('State')
ax.set_title('Rössler Driver System')
ax.legend()

plt.tight_layout()
plt.show()

# Discard initial transients
t_transient = 100.0
mask = sol.t >= t_transient

x_attractor = x[mask]
y_attractor = y[mask]
z_attractor = z[mask]

# 3D phase portrait
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_attractor, y_attractor, z_attractor, lw=0.5)

ax.set_xlabel(r'$x_D$')
ax.set_ylabel(r'$y_D$')
ax.set_zlabel(r'$z_D$')
ax.set_title('Rössler Chaotic Attractor')

ax.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

states = [x, y, z]
labels = [r'$x_D(t)$', r'$y_D(t)$', r'$z_D(t)$']

for i in range(3):

    ax[i].plot(sol.t, states[i], lw=0.8)
    ax[i].set_ylabel(labels[i])
    ax[i].grid(alpha=0.2)

ax[-1].set_xlabel('t')
fig.suptitle('Rössler Driver System: State Evolution')

plt.tight_layout()
plt.show()


#%% q5 part d

def rossler_coupled_system(t, state, coefs, gains):

    a, b, c = coefs
    kx, ky, kz = gains

    xD, yD, zD, xR, yR, zR = state

    # Synchronization errors
    ex = xR - xD
    ey = yR - yD
    ez = zR - zD

    # Control functions from Part (c)
    u1 = ey + ez - kx*ex
    u2 = -ex - (a + ky)*ey
    u3 = -zD*ex - (xR - c + kz)*ez

    # Driver system
    dxD = -yD - zD
    dyD = xD + a*yD
    dzD = b + zD*(xD - c)

    # Response system
    dxR = -yR - zR + u1
    dyR = xR + a*yR + u2
    dzR = b + zR*(xR - c) + u3

    return [dxD, dyD, dzD, dxR, dyR, dzR]


# Model parameters
a, b, c = 0.2, 0.2, 5.7
coefs = [a, b, c]

# Positive feedback gains
gains = [1.0, 1.0, 1.0]

# Different initial conditions
state0_driver = [1.0, 1.0, 1.0]
state0_response = [-3.0, 2.0, 5.0]

state0 = state0_driver + state0_response

# Numerical settings
t_span = (0.0, 150.0)
N = 12000
t_eval = np.linspace(*t_span, N)

sol = solve_ivp(
    fun=rossler_coupled_system,
    t_span=t_span,
    y0=state0,
    t_eval=t_eval,
    args=(coefs, gains),
    rtol=1e-9,
    atol=1e-11
)

if not sol.success:
    raise RuntimeError(sol.message)

# Extract Driver and Response trajectories
xD, yD, zD, xR, yR, zR = sol.y

# Synchronization errors
ex = xR - xD
ey = yR - yD
ez = zR - zD

error_norm = np.sqrt(ex**2 + ey**2 + ez**2)

fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

# Display initial synchronization process
mask = sol.t <= 30.0

states = [
    (xD, xR, 'x'),
    (yD, yR, 'y'),
    (zD, zR, 'z')
]

for i, (driver, response, label) in enumerate(states):

    ax[i].plot(sol.t[mask], driver[mask],
               'k-', lw=1.0, label='Driver')

    ax[i].plot(sol.t[mask], response[mask],
               'r--', lw=1.0, label='Response')

    ax[i].set_ylabel(label)
    ax[i].legend()

ax[-1].set_xlabel('t')
fig.suptitle('Rössler Driver–Response Synchronization')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))

mask = sol.t <= 25.0

ax.semilogy(sol.t[mask], np.maximum(np.abs(ex[mask]), 1e-14),
            label=r'$|e_x|$')

ax.semilogy(sol.t[mask], np.maximum(np.abs(ey[mask]), 1e-14),
            label=r'$|e_y|$')

ax.semilogy(sol.t[mask], np.maximum(np.abs(ez[mask]), 1e-14),
            label=r'$|e_z|$')

# Theoretical error norm for equal gains
error0 = error_norm[0]
error_theory = error0*np.exp(-sol.t[mask])

ax.semilogy(sol.t[mask], error_theory, 'k--',
            lw=1.5, label='Theoretical decay')

ax.set_xlabel('t')
ax.set_ylabel('Absolute synchronization error')
ax.set_title('Exponential Decay of Synchronization Errors')
ax.legend()

plt.tight_layout()
plt.show()

# Numerical verification
idx = np.argmin(np.abs(sol.t - 10.0))

print(f'Initial error norm: {error_norm[0]:.6e}')
print(f'Error norm at t = 10: {error_norm[idx]:.6e}')
print(f'Final error norm: {error_norm[-1]:.6e}')

# Remove synchronization transient
t_transient = 50.0
mask = sol.t >= t_transient

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(xD[mask], yD[mask], zD[mask],
        'k-', lw=0.6, label='Driver')

ax.plot(xR[mask], yR[mask], zR[mask],
        'r--', lw=0.6, alpha=0.7, label='Response')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
ax.set_title('Synchronized Rössler Attractors')

ax.view_init(elev=25, azim=-60)
ax.legend()

plt.tight_layout()
plt.show()

# Display the initial synchronization process
t_plot = 30.0
mask = sol.t <= t_plot

states = [
    (xD, xR, r'$x(t)$'),
    (yD, yR, r'$y(t)$'),
    (zD, zR, r'$z(t)$')
]

t_plot = 30.0
mask = sol.t <= t_plot

states = [
    (xD, xR, r'$x(t)$'),
    (yD, yR, r'$y(t)$'),
    (zD, zR, r'$z(t)$')
]

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, (driver, response, label) in enumerate(states):

    ax[i].plot(sol.t[mask], driver[mask], 'k-', lw=1.0, label='Driver')
    ax[i].plot(sol.t[mask], response[mask], 'r--', lw=1.0, label='Response')

    ax[i].set_ylabel(label)
    ax[i].legend()
    ax[i].grid(alpha=0.2)

ax[-1].set_xlabel('t')
fig.suptitle('Rössler Driver–Response Synchronization')

plt.tight_layout()
plt.show()