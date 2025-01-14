"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    def __init__(self, x_points, y_points):
        self.x_points = x_points
        self.y_points = y_points

    def area(self):
        n = len(self.x_points)
        if n <= 2:
            return float(0)
        # calculates the area of a polygon using the shoelace formula
        result = 0
        for i, (x, y) in enumerate(zip(self.x_points, self.y_points)):
            if i == len(self.x_points) - 1:
                break
            result += (self.x_points[i + 1] + x) * (self.y_points[i + 1] - y)
        return abs(result) / 2



class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass


    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        coordinates = contour(295)

        n = len(coordinates)
        # check is used to handle the case where the contour function returns fewer or equal to 2 points,
        # In this case, the shape is not closed and therefore has no area.
        if n <= 2:
            return float(0)
        area = 0
        for i in range(n):
            x1 = coordinates[i][0]
            y1 = coordinates[i][1]
            if i == n - 1:
                x2 = coordinates[0][0]
                y2 = coordinates[0][1]
            else:
                x2 = coordinates[i + 1][0]
                y2 = coordinates[i + 1][1]
            area += (x1 + x2) * (y1 -y2)
        area = area/2

        return np.float32(abs(area))


    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """
        before_sample = time.time()
        sample()
        after_sample = time.time()
        sampling_rate = 2 * maxtime / ((before_sample - after_sample) + 0.0005)
        num_pairs = int(sampling_rate)

        res_p = [sample() for _ in range(num_pairs)]
        center = np.average(res_p, axis=0)
        points_centered = res_p - center
        angles = np.arctan2(points_centered[:, 1], points_centered[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_points = [res_p[i] for i in sorted_indices]
        x_coor, y_coor = zip(*sorted_points)
        return MyShape(x_coor, y_coor)



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
