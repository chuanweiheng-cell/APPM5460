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

#%% q9 pb

mu = 0.25
rho = 10

def jacobian(x, y, mu, rho):
    J = np.array([
        [3*x**(1/2)/2 - 5*mu*x**(3/2)/2 - y/(2*x**(1/2)),
         -x**(1/2)],
        [rho*y, rho*(x - 1)]
    ])

    return J

# Equilibrium points
E1 = (0, 0)
E2 = (1/mu, 0)
E3 = (1, 1 - mu)

# Jacobians at equilibria with x > 0
J_E2 = jacobian(*E2, mu, rho)
J_E3 = jacobian(*E3, mu, rho)

print('Jacobian at E2:')
print(J_E2)

print('\nJacobian at E3:')
print(J_E3)

# Eigenvalues of the interior equilibrium
eigenvalues = np.linalg.eigvals(J_E3)

print('\nEigenvalues at E3:')
print(eigenvalues)
# %%
