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

#%% q1 

# Tokamak plasma dynamics

def tokamak(t, state, coefs):
    mu, rho = coefs
    u, log_y = state

    y = np.exp(log_y)

    du = (u**2 * (1 - mu*u**2) - y) / 2
    dlog_y = rho * (u**2 - 1)

    return [du, dlog_y]

# parameters

mu = 0.4
rho = 30.0

coefs = np.array([mu, rho], dtype=float)

# Initial conditions near E3 = (1, 1 - mu)
x0 = 1.05
y0 = 1 - mu

state0 = [np.sqrt(x0), np.log(y0)]

t_span = (0, 180)
N = 15000
t_eval = np.linspace(*t_span, N)

sol = solve_ivp(
    fun=tokamak,
    t_span=t_span,
    y0=state0,
    t_eval=t_eval,
    args=(coefs,),
    rtol=1e-9,
    atol=1e-11
)

if not sol.success:
    raise RuntimeError(sol.message)

u, log_y = sol.y

if np.any(u <= 0):
    raise RuntimeError('Solution reached the x = 0 boundary.')

# Recover original dimensionless variables
x = u**2
y = np.exp(log_y)


# Sawtooth oscillations: x(t)

plt.figure(figsize=(9, 4))

plt.plot(sol.t, x, label='x(t)')
plt.axhline(1, color='gray', linestyle='--',
            linewidth=1, label='Equilibrium x = 1')

plt.xlabel(r'$\tau$')
plt.ylabel(r'$x(\tau)$')
plt.title('Tokamak Plasma Temperature Oscillations')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# Sawtooth oscillations: steady periodic regime

mask = sol.t >= 100

plt.figure(figsize=(9, 4))

plt.plot(sol.t[mask], x[mask])

plt.xlabel(r'$\tau$')
plt.ylabel(r'$x(\tau)$')
plt.title('Sawtooth Oscillations After Initial Transient')
plt.tight_layout()
plt.show()


# Phase portrait: limit cycle

plt.figure(figsize=(7, 5))

plt.plot(x, y, linewidth=0.8, alpha=0.35,
         label='Full trajectory')

plt.plot(x[mask], y[mask], linewidth=1.5,
         label='Late-time trajectory')

plt.scatter(1, 1 - mu, color='red', s=45,
            zorder=5, label=r'$E_3=(1,1-\mu)$')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Tokamak Dynamics: Phase Portrait')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


#%% q2

# Rossler attractor

def rossler(t, state, coefs):
    a, b, c = coefs
    x, y, z = state

    dx = -y - z
    dy = x + a*y
    dz = b + z*(x - c)

    return [dx, dy, dz]


# Parameters
a, b, c = (0.2, 0.2, 5.7)
coefs = np.array([a, b, c], dtype=float)

# Initial conditions
state0 = np.array([1, 1, 1])

t_span = (0, 500)
N = 100000
t_eval = np.linspace(*t_span, N)

sol = solve_ivp(
    fun=rossler,
    t_span=t_span,
    y0=state0,
    t_eval=t_eval,
    args=(coefs,),
    rtol=1e-8,
    atol=1e-10
)

if not sol.success:
    raise RuntimeError(sol.message)

# Exclude initial transient
t_transient = 100
mask = sol.t >= t_transient

t = sol.t[mask]
x, y, z = sol.y[:, mask]


#%% plotting Rossler attractor: time series

plt.figure(figsize=(8, 4))
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.plot(t, z, label='z(t)')
plt.xlabel('t')
plt.ylabel('State')
plt.title('Rössler System: Time Series')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# Rossler attractor: 3D phase portrait

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, lw=0.5, color='m')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Rössler Attractor')

plt.tight_layout()
plt.show()


#%% Poincare section

# Define the Poincaré section y = 0, crossing in the negative direction
def poincare_section(t, state, coefs):
    return state[1]

poincare_section.direction = -1
poincare_section.terminal = False

# Integrate for sufficiently long to obtain at least 200 crossings
t_span = (0, 2500)
t_transient = 100

sol_poincare = solve_ivp(
    fun=rossler,
    t_span=t_span,
    y0=state0,
    args=(coefs,),
    events=poincare_section,
    rtol=1e-8,
    atol=1e-10,
    max_step=0.5
)

if not sol_poincare.success:
    raise RuntimeError(sol_poincare.message)

# Extract section crossings after the transient
t_cross = sol_poincare.t_events[0]
points = sol_poincare.y_events[0]

mask = (t_cross >= t_transient) & (points[:, 0] < 0)

t_cross = t_cross[mask]
x_n = points[mask, 0]
z_n = points[mask, 2]

print(f'Number of Poincaré intersections: {len(x_n)}')

if len(x_n) < 200:
    raise RuntimeError('Fewer than 200 intersections. Increase t_span.')

# Plot the Poincaré section in the x-z plane
plt.figure(figsize=(7, 5))
plt.scatter(x_n, z_n, s=8)

plt.xlabel(r'$x_n$')
plt.ylabel(r'$z_n$')
plt.title(r'Rössler Attractor: Poincaré Section $\Sigma$')
plt.tight_layout()
plt.show()


#%%  Poincare return map

plt.figure(figsize=(7, 5))
plt.scatter(x_n[:-1], x_n[1:], s=8)

plt.xlabel(r'$x_n$')
plt.ylabel(r'$x_{n+1}$')
plt.title('Rössler Attractor: Poincaré Return Map')
plt.tight_layout()
plt.show()


#%% Polynomial approximation of the Poincaré map

from numpy.polynomial import Polynomial

# Polynomial degrees
degree_P = 5
degree_Q = 5

# Fit polynomials using least squares
P = Polynomial.fit(x_n[:-1], x_n[1:], deg=degree_P)
Q = Polynomial.fit(x_n, z_n, deg=degree_Q)

# Evaluate polynomial approximations
x_fit = np.linspace(np.min(x_n), np.max(x_n), 1000)

P_fit = P(x_fit)
Q_fit = Q(x_fit)

# Compute root mean squared errors
rmse_P = np.sqrt(np.mean((x_n[1:] - P(x_n[:-1]))**2))
rmse_Q = np.sqrt(np.mean((z_n - Q(x_n))**2))

print(f'RMSE for P(x): {rmse_P:.6e}')
print(f'RMSE for Q(x): {rmse_Q:.6e}')

# Print polynomial coefficients in standard power basis
print('\nP(x) coefficients:')
print(P.convert().coef)

print('\nQ(x) coefficients:')
print(Q.convert().coef)


# Polynomial approximation: z_n = Q(x_n)

plt.figure(figsize=(7, 5))

plt.scatter(x_n, z_n, s=8, label='Poincaré data')
plt.plot(x_fit, Q_fit, color='red', linewidth=1.5,
         label=f'Polynomial fit (degree {degree_Q})')

plt.xlabel(r'$x_n$')
plt.ylabel(r'$z_n$')
plt.title(r'Polynomial Approximation: $z_n \approx Q(x_n)$')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# Polynomial approximation: x_(n+1) = P(x_n)

plt.figure(figsize=(7, 5))

plt.scatter(x_n[:-1], x_n[1:], s=8, label='Poincaré data')
plt.plot(x_fit, P_fit, color='red', linewidth=1.5,
         label=f'Polynomial fit (degree {degree_P})')

plt.xlabel(r'$x_n$')
plt.ylabel(r'$x_{n+1}$')
plt.title(r'Polynomial Approximation: $x_{n+1} \approx P(x_n)$')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


#%% Periodic orbits from the Poincaré map

from scipy.optimize import brentq, root

x_min, x_max = np.min(x_n), np.max(x_n)

# Iterate the fitted Poincaré map k times
def iterate_P(x, k):
    values = [float(x)]

    for _ in range(k):
        values.append(float(P(values[-1])))

    return np.array(values)


# Define the equation P^(k)(x) - x = 0
def residual(x, k):
    values = iterate_P(x, k)

    # Reject iterations outside the fitted data range
    if not np.all(np.isfinite(values)):
        return np.nan

    if np.any((values < x_min) | (values > x_max)):
        return np.nan

    return values[-1] - x


# Search for roots of P^(k)(x) = x
poly_orbits = []
x_grid = np.linspace(x_min, x_max, 10001)

for k in [1, 2, 3]:

    f_grid = np.array([residual(x, k) for x in x_grid])
    roots = []

    for i in range(len(x_grid) - 1):

        if not np.isfinite(f_grid[i:i+2]).all():
            continue

        if f_grid[i] * f_grid[i+1] < 0:
            x_root = brentq(
                lambda x: residual(x, k),
                x_grid[i], x_grid[i+1]
            )

            if not any(abs(x_root - r) < 1e-5 for r in roots):
                roots.append(x_root)

    for x_root in roots:

        # Exclude roots with a smaller fundamental period
        if any(abs(iterate_P(x_root, j)[-1] - x_root) < 1e-5
               for j in range(1, k)):
            continue

        orbit_x = iterate_P(x_root, k)[:-1]

        # Remove cyclic duplicates of the same orbit
        duplicate = any(
            item['k'] == k and
            np.allclose(np.sort(orbit_x),
                        np.sort(item['orbit_x']),
                        atol=1e-4, rtol=0)
            for item in poly_orbits
        )

        if duplicate:
            continue

        poly_orbits.append({
            'k': k,
            'orbit_x': orbit_x,
            'x0': x_root,
            'z0': float(Q(x_root))
        })


#  Refine periodic orbits using the original ODEs

# Detect crossings of y = 0 with negative direction
def section_event(t, state, coefs):
    return state[1]

section_event.direction = -1
section_event.terminal = False


# Advance from one Poincaré crossing to the next
def next_crossing(xz):
    x, z = xz

    sol = solve_ivp(
        fun=rossler,
        t_span=(0, 12),
        y0=[x, 0, z],
        args=(coefs,),
        events=section_event,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.2
    )

    # Exclude the crossing at the initial time t = 0
    indices = np.where(sol.t_events[0] > 1e-5)[0]

    if not sol.success or len(indices) == 0:
        raise RuntimeError('Poincare return not found.')

    i = indices[0]

    return sol.y_events[0][i, [0, 2]], sol.t_events[0][i]


# Apply the full Poincaré map k times
def full_return(xz, k):
    point = np.asarray(xz, dtype=float)
    period = 0.0

    for _ in range(k):
        point, dt = next_crossing(point)
        period += dt

    return point, period


refined_orbits = []

for item in poly_orbits:
    k = item['k']
    guess = [item['x0'], item['z0']]

    # Solve full_return(x,z,k) = (x,z)
    result = root(
        lambda xz: full_return(xz, k)[0] - xz,
        guess,
        tol=1e-10
    )

    xz = result.x
    final, period = full_return(xz, k)
    error = np.linalg.norm(final - xz)

    if error > 1e-7:
        print(f'Period {k}: refinement failed.')
        continue

    refined_orbits.append({
        'k': k,
        'x0': xz[0],
        'z0': xz[1],
        'period': period,
        'error': error
    })

    print(f'\nPeriod {k} orbit:')
    print(f'  Initial condition: ({xz[0]:.8f}, 0, {xz[1]:.8f})')
    print(f'  Continuous-time period: {period:.8f}')
    print(f'  Return error: {error:.3e}')


#%%  Periodic orbits in 3D phase space

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

for i, orbit in enumerate(refined_orbits):

    k = orbit['k']
    period = orbit['period']

    # Initial condition on the Poincaré section
    state0 = [orbit['x0'], 0, orbit['z0']]

    # Integrate over one complete period
    t_span = (0, period)
    t_eval = np.linspace(*t_span, 3000)

    sol = solve_ivp(
        fun=rossler,
        t_span=t_span,
        y0=state0,
        t_eval=t_eval,
        args=(coefs,),
        rtol=1e-10,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    x, y, z = sol.y

    # Plot the periodic orbit
    
    ax.plot(
        x, y, z,
        lw=1.5,
        label=(f'Orbit {i+1}: Period {k}\n'
            f'IC = ({orbit["x0"]:.5f}, 0, {orbit["z0"]:.5f})')
    )


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Periodic Orbits in the Rössler Attractor')
ax.legend(frameon=False)

plt.tight_layout()
plt.show()

