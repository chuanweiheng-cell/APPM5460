#%% imports

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

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


#%% q4 part c

lmbd_0 = 2.0 / jnp.pi

x = jnp.linspace(-1.0, 1.0, 100)


# Integral for ONE value of x
def integral_at_x(x_i, func_distribution, y_range):

    integrand = (
        func_distribution
        / (1.0 + (x_i - y_range) ** 2.0)
    )

    return jnp.trapezoid(
        integrand,
        y_range
    )


# Vectorized integral over all x values
integral_all_x = jax.vmap(
    integral_at_x,
    in_axes=(0, None, None)
)


# Operator T
def operator_T(func_distribution, x, lmbd):

    y_range = x

    integral_values = integral_all_x(
        x,
        func_distribution,
        y_range
    )

    return (
        jnp.sin(2.0 * jnp.pi * x)
        + lmbd * integral_values
    )


# Fixed-point iteration
def fixed_point_iteration(
    f_initial,
    x,
    lmbd,
    tol=1e-8,
    max_iter=1000
):

    f = f_initial

    error_history = []

    for count in range(max_iter):

        f_new = operator_T(
            f,
            x,
            lmbd
        )

        # Sup-norm difference between successive iterates
        error = jnp.max(
            jnp.abs(f_new - f)
        )

        error_history.append(error)

        f = f_new

        if error < tol:

            return (
                f,
                jnp.array(error_history),
                count + 1,
                True
            )

    return (
        f,
        jnp.array(error_history),
        max_iter,
        False
    )


#Initial states
f_initial_0 = jnp.zeros_like(x)

f_initial_1 = x

f_initial_2 = jnp.exp(x)

initial_states = [
    f_initial_0,
    f_initial_1,
    f_initial_2
]

initial_labels = [
    r'$f_0(x)=0$',
    r'$f_0(x)=x$',
    r'$f_0(x)=e^x$'
]


#%Lambda values


lmbd_values = jnp.linspace(0.05 * lmbd_0, 1.1 * lmbd_0, 7)


#Run experiments
tol = 1e-8
max_iter = 1000

# Loop over each initial function
for f_initial, function_label in zip(
    initial_states,
    initial_labels
):

    plt.figure(figsize=(8, 5))

    print('\n========================================')
    print(f'Initial state: {function_label}')
    print('========================================')

    # Sweep through lambda values
    for lmbd in lmbd_values:

        f_final, error_history, count, converged = fixed_point_iteration(
            f_initial,
            x,
            lmbd,
            tol=tol,
            max_iter=max_iter
        )

        print(
            f'lambda = {lmbd:.6f}: '
            f'iterations = {count}, '
            f'converged = {converged}'
        )

        plt.semilogy(
            jnp.arange(1, len(error_history) + 1),
            error_history,
            'o-',
            markersize=3,
            label=rf'$\lambda={lmbd:.3f}$'
        )

    plt.xlabel('Iteration')
    plt.ylabel(r'$\|f_{k+1}-f_k\|_\infty$')
    plt.title(
        rf'Fixed-point convergence for {function_label}'
    )
    plt.legend(frameon=False)

    plt.show()
    
#%% q5

def W(x, N_terms=1000, alpha=0.5):

    n = jnp.arange(N_terms)

    terms = (2.0 ** (-alpha * n) * jnp.cos(2.0 ** n * x))

    return jnp.sum(terms)

W_vec = jax.vmap(W)

x_vals = jnp.linspace(-jnp.pi, jnp.pi, 10000)

W_vals = W_vec(x_vals)

plt.plot(x_vals, W_vals)
plt.title('Weierstrass function')
plt.show()