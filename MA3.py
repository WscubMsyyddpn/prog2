""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import os
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):
    x_inside, y_inside = [], []
    x_outside, y_outside = [], []

    for _ in range(n):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)

        #x^2 + y^2 <= 1 (inside circle)
        if (x**2 + y**2) <= 1:
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    n_c = len(x_inside)
    pi_approx = 4.0 * n_c / n
    print(f"Approximation: {pi_approx:.6f}")
    point_size = max(0.5, min(10.0, 10000.0 / n))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_outside, y_outside, color='blue', s=point_size, label=f'Outside ({n - n_c})')
    ax.scatter(x_inside, y_inside, color='red', s=point_size, label=f'Inside ({n_c})')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(f"Monte Carlo pi (n = {n:,}) ≈ {pi_approx:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", markerscale=max(1, int(15 / point_size)))

    plt.show()

    return pi_approx

# Exc2, approximation
def sphere_volume(n, d): 
    # generate n d-dimensional points in [-1, 1]^d
    points = [[random.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(n)]
    
    # map to compute sum of squared coordinates: sum(x_i^2)
    squared_sums = map(lambda pt: sum(x**2 for x in pt), points)
    
    # Concept 3: Higher-order function `filter` to find points satisfying sum(x_i^2) <= 1.0
    inside_points = list(filter(lambda sq_dist: sq_dist <= 1.0, squared_sums))
    
    # Volume of bounding hypercube [-1, 1]^d is 2^d
    hypercube_volume = 2.0 ** d
    
    # Approximated hypersphere volume
    return (len(inside_points) / n) * hypercube_volume


#Exc2, real value
def hypersphere_exact(n, d):
    # r=1
    return (m.pi ** (d / 2.0)) / m.gamma(d / 2.0 + 1)


#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    n_c = 0
    for _ in range(n):
        sum_square = 0.0
        for _ in range(d):
            coord = random.uniform(-1.0,1.0)
            sum_square = coord * coord
        if sum_square <= 1:
            n_c += 1
    return (2.0**d) *(n_c/n)

#Exc4: parallel code - parallelize actual computations by splitting data

def _woker_function(n,d):
    # generate n d-dimensional points in [-1, 1]^d
        points = [[random.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(n)]
        squared_sums = map(lambda pt: sum(x**2 for x in pt), points)
        inside_points = list(filter(lambda sq_dist: sq_dist <= 1.0, squared_sums))
        
        # Approximated hypersphere volume
        return (len(inside_points))

def sphere_volume_parallel(n, d, num_workers=4):
    # Divide n into chunks to distribute across CPU cores
    chunk_size = n // num_workers
    remainder = n % num_workers
    chunks = [chunk_size + 1 if i < remainder else chunk_size for i in range(num_workers)]
    
    # ProcessPoolExecutor run in a context manager as shown in the text
    with future.ProcessPoolExecutor() as ex:
        # Map the worker task across the chunks, passing d as well
        results = ex.map(_woker_function, chunks, [d] * num_workers)
        total_inside = sum(results)

    hypercube_volume = 2.0 ** d
    return (total_inside / n) * hypercube_volume
    
def main():
    # Exc1
    # dots = [1000, 10000, 100000]
    # for n in dots:
    #     approximate_pi(n)

    # # Exc2
    # n = 100000
    # d = 2
    # sphere_volume(n, d)
    # print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # n = 100000
    # d = 11
    # sphere_volume(n, d)
    # print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # # Exc3
    # n = 1000000
    # d = 11
    # start = pc()
    # sphere_volume(n, d)
    # stop = pc()
    # print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    # print("What is numba time?")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    print(sphere_volume(n, d))
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")
    start = pc()
    print(sphere_volume_parallel(n,d,4))
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")


if __name__ == '__main__':
	main()
