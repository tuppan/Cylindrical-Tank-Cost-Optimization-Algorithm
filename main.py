import numpy as np
import matplotlib.pyplot as plt
import random as rd

alpha = 0.001
mu = 1e6
delta = 1e-5
tol = 1e-6

def cost_fcn(r, h):
    return 350 * (np.pi * r ** 2) + 100 * (2 * np.pi * r * h)
"""
Cost function for the cylinder
Base: 150 / m²
Lateral surface: 100 / m^2
Top: 200 / m²
Cost = 150 (pi * r²) + 100 (2 * pi * r * h) + 200 (pi * r²)
Simplified to: Cost = 350 (pi * r²) + 100 (2 * pi * r * h)
"""
def restriction_fcn(r, h):
    volume = np.pi * (r ** 2) * h
    return 500 - volume
"""
Restriction function for the cylinder
500 - (pi * r² * h) <= 0
Cylinder volume must be less than or equal to 500 m³
"""
def penalization_fcn(r,h, mu):
    f = cost_fcn(r,h)
    g = restriction_fcn(r,h)
    return f + mu * (max(0, g))**2
"""
Penalization function
If the restriction isn't satisfied, the penalization function will add a penalty to the cost function
"""
def partial_r_derivative (r, h, mu, delta):
    return (penalization_fcn(r + delta, h, mu) - penalization_fcn(r, h, mu)) / (delta)
"""
Partial derivative in respect to r
If we freeze the height and give a short step to the radius, does the cost go up or down? And how fast?
"""
def partial_h_derivative (r, h, mu, delta):
    return (penalization_fcn(r, h + delta, mu) - penalization_fcn(r, h, mu)) / (delta)
"""
Partial derivative in respect to h
If we freeze the radius and give a short step to the height, does the cost go up or down? And how fast?
"""
def random_start():
    r = rd.uniform(0.1, 30)
    h = rd.uniform(0.1, 15)
    return r, h
# Generate random values for r and h to start the optimization process.
def gradient_method():
    r, h = random_start()
    for i in range (1000000):
        dr = partial_r_derivative(r, h, mu, delta)
        dh = partial_h_derivative(r, h, mu, delta)
        if np.sqrt(dr**2 + dh**2) < tol:
            break
        # Code break if the gradient vector magnitude is low enough, meaning we are close to a local minimum
        r_new = r - alpha * dr
        h_new = h - alpha * dh
        r = r_new
        h = h_new
        # In each iteration k, the tank dimensions are updated by subtracting a piece of the corresponding derivative
    return cost_fcn(r, h), r, h

TODO: Add the plotting function and main function
