import numpy as np
import matplotlib.pyplot as plt
import random as rd

delta = 1e-6
tol = 1e-7
temp1 = 0.0
temp2 = 0.0

def cost_fcn(r, h):
    return 350 * (np.pi * r ** 2) + 100 * (2 * np.pi * r * h)
"""
Cost function for the cylinder:
Base: 150 / m²
Lateral surface: 100 / m²
Top: 200 / m²
Cost: f(r, h)= 150 (pi * r²) + 100 (2 * pi * r * h) + 200 (pi * r²)
Simplified to: f(r, h) = 350 (pi * r²) + 100 (2 * pi * r * h)
"""
def restriction_fcn(r, h):
    volume = np.pi * (r ** 2) * h
    return 500 - volume
"""
Restriction function for the cylinder:
g(r, h) = 500 - (pi * r² * h) <= 0
Cylinder volume must be less than or equal to 500 m³.*
"""
def penalization_fcn(r,h, mu):
    f = cost_fcn(r,h)
    g = restriction_fcn(r,h)
    return f + mu * (max(0, g))**2
"""
Penalization function:
If the restriction isn't satisfied, the penalization function will add a penalty to the cost function.
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
    r = rd.uniform(0.1, 12)
    h = rd.uniform(0.1, 24)
    return r, h
# Generate random values for r and h to start the optimization process.
def gradient_method():
    global mu, alpha
    mu = 10.0
    alpha = float(1e-6)
    r, h = random_start()
    count = 0
    temp1 = r
    temp2 = h
    for i in range (1000000):
    # The optimization process will run for a maximum of a million iterations.
        if i % 1000 == 0 and i > 0:
        # Every a thousand iterations, the alpha coefficient is reduced by a small amount.
            alpha *= 0.95
            if restriction_fcn(r, h) > 0:
            # If g(r, h) > 0, the penalty coefficient mu is doubled to increase the penalty for violating the restriction.
                mu *= 2
 
        dr = partial_r_derivative(r, h, mu, delta)
        dh = partial_h_derivative(r, h, mu, delta)
        # The partial derivatives of the penalization function with respect to r and h are calculated for each iteration.
        r_new = r - alpha * dr
        h_new = h - alpha * dh
        # The new values of r and h are recalculated by subtracting a fraction of the corresponding derivative.
        r_new = max(0.1, r_new)
        h_new = max(0.1, h_new)
        # The new values of r and h are constrained to be at least 0.1 to avoid negative or zero dimensions.
        if abs(r_new - r) < tol and abs(h_new - h) < tol:
            break
        # Code break if the absolute difference between the new and old values of r and h is less than the tolerance level.
        r = r_new
        h = h_new
        count += 1
        # The values of r and h are updated for the next iteration, and the iteration count is incremented.
        
    return cost_fcn(r, h), r, h, count, temp1, temp2

def main():
    reps = int(input("Number of iterations: "))
    r_array = []
    h_array = []
    cost_array = []
    count_array = []
    # The vectors are initialized.
    explosion_count = 0
    # The explosion_count variable is initialized to count the number of times the cost exceeds a certain threshold.
    print("Iteration || Initial (r, h) ||   Cost   || Final (r, h) || Final Volume || Iterations")
    for i in range(reps):
        cost, r, h, count, temp1, temp2 = gradient_method()
        r_array.append(r)
        h_array.append(h)
        cost_array.append(cost)
        # The results of the optimization process are stored in the corresponding vectors.
        if cost > 42100:
            explosion_count += 1
        # If the cost exceeds 42100, the explosion_count variable is incremented.
        count_array.append(count)
        # The number of iterations taken for the optimization process is stored in the count_array vector.
        print(f"    {i+1}     ||  ({temp1:.2f}; {temp2:.2f}) || {cost_array[-1]:.2f} || ({r_array[-1]:.2f}; {h_array[-1]:.2f}) ||    {np.pi * r_array[-1]**2 * h_array[-1]:.2f}    || {count_array[-1]}")
        print("-"*81)
        # The initial randomized and the optimized values, and cost are printed for each iteration.
    print(f"Average cost = {np.mean(cost_array):.4f}")
    print(f"Minimum cost = {np.min(cost_array):.4f}")
    print(f"Max cost = {np.max(cost_array):.4f}")
    print(f"Average (r, h) = ({np.mean(r_array):.2f}; {np.mean(h_array):.2f})")
    print(f"Best (r, h) = ({r_array[np.argmin(cost_array)]:.2f}; {h_array[np.argmin(cost_array)]:.2f})")
    print(f"Explosions: {explosion_count}")
    # The average, minimum, and maximum costs, as well as the average and the optimal values of r and h, are printed.

if True:
    main()