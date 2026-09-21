#%% imports

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
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


#%% q6

# ODE system
def sys(t, state):

    x, y = state

    dx = -y + x * (1 - x ** 2 - y ** 2)
    dy = x + y * (1 - x ** 2 - y ** 2)

    return [dx, dy]


# Nullcline functions
def dxdt(x, y):

    return -y + x * (1 - x ** 2 - y ** 2)


def dydt(x, y):

    return x + y * (1 - x ** 2 - y ** 2)


# Solver parameters
t_span = (0, 100)
N = 1000
t_eval = jnp.linspace(*t_span, N)

initial_conditions = [
    [0.2, 0],
    [0.5, 0],
    [1.0, 0],
    [1.5, 0],
    [2.0, 0]
]


#%% plotting

plt.figure(figsize=(8, 8))

# Phase trajectories
for i, state0 in enumerate(initial_conditions):

    sol = solve_ivp(
        fun=sys,
        t_span=t_span,
        y0=state0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    x_sol = sol.y[0]
    y_sol = sol.y[1]

    plt.plot(x_sol, y_sol, label=f'I.C.: {initial_conditions[i]}')


# Nullcline grid
x_grid = jnp.linspace(-2, 2, 600)
y_grid = jnp.linspace(-2, 2, 600)

X, Y = jnp.meshgrid(x_grid, y_grid)

# x-nullcline: dx/dt = 0
c_x=plt.contour(
    X, Y, dxdt(X, Y),
    levels=[0],
    linestyles='--'
)

# y-nullcline: dy/dt = 0
c_y=plt.contour(
    X, Y, dydt(X, Y),
    levels=[0],
    linestyles=':'
)

plt.clabel(c_x)

plt.clabel(c_y)

# Formatting
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Phase Portrait and Nullclines')

plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.legend(frameon=False)
plt.axis('equal')
plt.grid()

plt.savefig('homework/hw1/q6.png')
plt.show()