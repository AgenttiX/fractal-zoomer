# Created by Mika Mäki, 2018
# for Tampere University of Technology course
# RAK-19006 Python 3 for scientific computing

import numba
import numba.cuda as cuda
import numpy as np

import typing as tp
import math


# Constants for CUDA array creation
block_dim = (32, 8)
grid_dim = (32, 16)


# CUDA core functions
# (one instance <-> one CUDA core <-> one fractal pixel)

@cuda.jit(device=True)
def frac_mandel(x, y, max_iter):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iter):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return i

    return max_iter


@cuda.jit(device=True)
def frac_mandel_color(x, y, max_iter):
    c = complex(x, y)
    z = 0.0j
    esc = False

    for i in range(max_iter):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            esc = True
            break

    if esc:
        # Continuous indexing for smoother coloring
        # http://www.paridebroggi.com/2015/05/fractal-continuous-coloring.html
        ind = i + 1 - (math.log(2.0) / abs(z)) / math.log(2.0)
        return \
            math.sin(0.016 * ind + 4) * 230 + 25, \
            math.sin(0.013 * ind + 2) * 230 + 25, \
            math.sin(0.01  * ind + 1) * 230 + 25
    else:
        return 0, 0, 0


@cuda.jit(device=True)
def frac_julia(x, y, c, max_iter):
    z = complex(x, y)
    for i in range(max_iter):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return i
    return max_iter


@cuda.jit(device=True)
def frac_julia_color(x, y, c, max_iter):
    z = complex(x, y)
    esc = False

    for i in range(max_iter):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            esc = True
            break

    if esc:
        # Continuous indexing for smoother coloring
        # http://www.paridebroggi.com/2015/05/fractal-continuous-coloring.html
        ind = i + 1 - (math.log(2.0) / abs(z)) / math.log(2.0)
        return \
            math.sin(0.016 * ind + 4) * 230 + 25, \
            math.sin(0.013 * ind + 2) * 230 + 25, \
            math.sin(0.01 * ind + 1) * 230 + 25
    else:
        return 0, 0, 0


@cuda.jit(device=True)
def frac_carpet(x, y):
    while x > 0 or y > 0:
        if x % 3 == 1 and y % 3 == 1:
            return False
        x //= 3
        y //= 3
    return True


# CUDA kernels
# These initialize the CUDA cores for computation
# Based on
# https://github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb

@cuda.jit
def kernel_mandel(x_min, x_max, y_min, y_max, image, max_iter):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (x_max - x_min) / width
    pixel_size_y = (y_max - y_min) / height

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, width, grid_x):
        real = x_min + x * pixel_size_x
        for y in range(start_y, height, grid_y):
            imag = y_min + y * pixel_size_y
            image[y, x] = frac_mandel(real, imag, max_iter)


@cuda.jit
def kernel_mandel_color(x_min, x_max, y_min, y_max, image, max_iter):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (x_max - x_min) / width
    pixel_size_y = (y_max - y_min) / height

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, width, grid_x):
        real = x_min + x * pixel_size_x
        for y in range(start_y, height, grid_y):
            imag = y_min + y * pixel_size_y
            image[y, x, :] = frac_mandel_color(real, imag, max_iter)


@cuda.jit
def kernel_julia(x_min, x_max, y_min, y_max, image, max_iter, c):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (x_max - x_min) / width
    pixel_size_y = (y_max - y_min) / height

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, width, grid_x):
        real = x_min + x * pixel_size_x
        for y in range(start_y, height, grid_y):
            imag = y_min + y * pixel_size_y
            image[y, x] = frac_julia(real, imag, c, max_iter)


@cuda.jit
def kernel_julia_color(x_min, x_max, y_min, y_max, image, max_iter, c):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (x_max - x_min) / width
    pixel_size_y = (y_max - y_min) / height

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, width, grid_x):
        real = x_min + x * pixel_size_x
        for y in range(start_y, height, grid_y):
            imag = y_min + y * pixel_size_y
            image[y, x, :] = frac_julia_color(real, imag, c, max_iter)


@cuda.jit
def kernel_carpet(image):
    size = image.shape[0]

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, size, grid_x):
        for y in range(start_y, size, grid_y):
            image[y, x] = frac_carpet(x, y)


# Wrappers
# These wrap the CUDA kernels for simpler access from normal Python code

def mandel(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: tp.Tuple[int, int],
        max_iter: int):
    image = np.zeros(resolution, dtype=np.uint32)
    device_image = cuda.to_device(image)
    kernel_mandel[grid_dim, block_dim](x_min, x_max, y_min, y_max, device_image, max_iter)
    device_image.to_host()
    return image


def mandel_color(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: tp.Tuple[int, int],
        max_iter: int):
    image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
    device_image = cuda.to_device(image)
    kernel_mandel_color[grid_dim, block_dim](x_min, x_max, y_min, y_max, device_image, max_iter)
    device_image.to_host()
    return image


def julia(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: tp.Tuple[int, int],
        max_iter: int,
        c: complex):
    image = np.zeros(resolution, dtype=np.uint32)
    device_image = cuda.to_device(image)
    kernel_julia[grid_dim, block_dim](x_min, x_max, y_min, y_max, device_image, max_iter, c)
    device_image.to_host()
    return image


def julia_color(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: tp.Tuple[int, int],
        max_iter: int,
        c: complex):
    image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
    device_image = cuda.to_device(image)
    kernel_julia_color[grid_dim, block_dim](x_min, x_max, y_min, y_max, device_image, max_iter, c)
    device_image.to_host()
    return image


def carpet(size: int):
    image = np.ones((3**size, 3**size), dtype=bool)
    device_image = cuda.to_device(image)
    kernel_carpet[grid_dim, block_dim](device_image)
    device_image.to_host()
    return image


# Utility functions

@numba.jit
def color(image: np.array, max_iter: int=0):
    """
    Color a fractal on CPU
    :param image: 2D Numpy array
    :param max_iter: maximum iteration used in the fractal rendering (needed to set these as black)
    :return: a colored fractal - Numpy 3D array (RGB 0-255)
    """

    # Hervanta constants for trigonometric functions taken from
    # http://www.paridebroggi.com/2015/05/fractal-continuous-coloring.html

    colored = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == max_iter:
                colored[y, x, :] = (0, 0, 0)
            else:
                colored[y, x, 0] = math.sin(0.016 * image[y, x] + 4) * 230 + 25
                colored[y, x, 1] = math.sin(0.013 * image[y, x] + 2) * 230 + 25
                colored[y, x, 2] = math.sin(0.01 * image[y, x] + 1) * 230 + 25

    return colored
