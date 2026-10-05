""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
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
    # create random points inside a hyper cube
    points = [[random.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(n)]

    # lambdafunction squares
    square_func = lambda pt: sum(x**2 for x in pt)
    # applies the function to the interable points
    square = map(square_func, points)

    # criteria: squared distance is less then 1.0 which means inside unit sphere
    filter_func = lambda sq_dist: sq_dist <= 1.0
    # applies the filter on square based on criteria from filter_func
    points_inside = list(filter(filter_func, square))
    
    # volume of hypercube
    volume_hc = 2.0 ** d
    
    # approximated hypersphere volume
    return (len(points_inside) / n) * volume_hc


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
            sum_square += coord * coord
        if sum_square <= 1:
            n_c += 1
    return (n_c/n) * (2.0**d)

#Exc4: parallel code - parallelize actual computations by splitting data

def _worker_function(n,d):
    points = [[random.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(n)]
    square_func = lambda pt: sum(x**2 for x in pt)
    square = map(square_func, points)
    filter_func = lambda sq_dist: sq_dist <= 1.0
    points_inside = list(filter(filter_func, square))
        
    # return number of inside points
    return (len(points_inside))

def sphere_volume_parallel(n, d, num_workers):
    # divide into equal worker chunks
    chunk_size = n // num_workers
    # remainder chunk
    remainder = n % num_workers
    # distrubution of chunks and remainder (gives all workers a chunk or chunk + remainder)
    chunks = [chunk_size + 1 if i < remainder else chunk_size for i in range(num_workers)]

    
    with future.ProcessPoolExecutor() as ex:
        # map worker task across the chunks, and dimension d. 
        results = ex.map(_worker_function, chunks, [d] * num_workers)
        total_inside = sum(results)

    hypercube_volume = 2.0 ** d
    return (total_inside / n) * hypercube_volume
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print(f"Approximation volume of {d} dimentional sphere = {sphere_volume(n, d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print(f"Approximation volume of {d} dimentional sphere = {sphere_volume(n, d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")

    print("What is numba time?")
    start = pc()
    sphere_volume_numba(n, d)
    stop = pc()
    print(f"Exc3: Numba time of {d} and {n}: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    
    print("What is parallel time?")
    start = pc()
    sphere_volume_parallel(n,d,6)
    stop = pc()
    print(f"Exc4: Parallel time of {d} and {n}: {stop-start}")


if __name__ == '__main__':
	main()



'''
VSCODE
Approximation: 3.224000
Approximation: 3.139600
Approximation: 3.129640
Actual volume of 2 dimentional sphere = 3.141592653589793
Approximation volume of 2 dimentional sphere = 3.13952
Actual volume of 11 dimentional sphere = 1.8841038793898994
Approximation volume of 11 dimentional sphere = 2.08896
Exc3: Sequential time of 11 and 1000000: 5.507090199971572
What is numba time?
Exc3: Sequential time of 11 and 1000000: 1.6331422999501228
Exc4: Sequential time of 11 and 1000000: 5.62280029989779
What is parallel time?
Exc4: Sequential time of 11 and 1000000: 3.4570555000100285

LINUX
Approximation: 3.088000
Approximation: 3.152400
Approximation: 3.141680
Actual volume of 2 dimentional sphere = 3.141592653589793
Approximation volume of 2 dimentional sphere = 3.13428
Actual volume of 11 dimentional sphere = 1.8841038793898994
Approximation volume of 11 dimentional sphere = 1.96608
Exc3: Sequential time of 11 and 1000000: 13.917332861048635
What is numba time?
Exc3: Sequential time of 11 and 1000000: 1.8035552239743993
Exc4: Sequential time of 11 and 1000000: 13.836500756966416
What is parallel time?
Exc4: Sequential time of 11 and 1000000: 2.86912150104763




'''