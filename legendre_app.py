import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st
import math

def legendre_ode(x, y, n):
    """
    Legendre's differential equation:
    (1-x^2)y'' - 2xy' + n(n+1)y = 0
    
    Rewritten as a system of first-order ODEs:
    y[0]' = y[1]
    y[1]' = (2xy[1] - n(n+1)y[0]) / (1-x^2)
    """
    return [
        y[1],
        (2*x*y[1] - n*(n+1)*y[0]) / (1-x**2)
    ]   


def solve_legendre(n, x_range=(-1, 1)):
    """
    Solve Legendre's equation for given n over x_range.
    Initial conditions: p(1) = p'(1) = 1
    """
    # Solve backwards from x=1 to x=-1
    sol = solve_ivp(
        lambda x, y: legendre_ode(x, y, n),
        t_span=[0.999, -1],
        y0=[1, 1],  # Initial conditions: p(1) = p'(1) = 1
        dense_output=True
    )
    # Generate solution over desired x range
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = sol.sol(x)
    return x, y[0]

if __name__ == "__main__":

    st.markdown("## Solve Legendre's differential equation:")
    st.latex(r"(1 - x^2) \frac{d^2 y}{dx^2} - 2x \frac{dy}{dx} + n(n+1)y = 0")
    n = st.slider("Enter n", min_value=0., max_value=5., step=0.05)

    x, y = solve_legendre(n)
    y[0] = 1 if n == int(n) else 1e3
    y[0] *= -1 if math.ceil(n) % 2 == 1 else 1
        
    fig, ax = plt.subplots()
    ax.plot(x, y, label=f'n={n}')
    ax.set_ylim([-2, 2])
    ax.scatter([1], [1], s=15, c='C0')
    if n == int(n):
        ax.scatter([-1], [y[0]], s=15, c='C0')

    st.pyplot(fig)
        