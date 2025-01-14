"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass


    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        polynom_degree = int(d / 3)
        if polynom_degree > 12:
            polynom_degree = 12
        xs = []
        ys = []

        start_time = time.perf_counter()

        while True:
            new_x = random.uniform(a, b)
            new_y = f(new_x)

            xs.append(new_x)
            ys.append(new_y)

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time

            if elapsed_time >= maxtime * (1 / (polynom_degree + 1)):
                break

        xs = np.array(xs)
        ys = np.array(ys)

        A = np.array([[x ** i for i in range(d + 1)] for x in xs])
        m = A.shape[0]
        n = A.shape[1]
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        for i in range(n):
            v = A[:, i]
            for j in range(i):
                R[j, i] = Q[:, j].dot(A[:, i])
                v = v - R[j, i] * Q[:, j]
            R[i, i] = np.linalg.norm(v)
            Q[:, i] = v / R[i, i]
        B = Q.T @ ys
        coefficients = np.linalg.inv(R) @ B  # compute the linear system Ax=b

        def polynomial(x):
            return sum([coefficients[i] * x ** i for i in range(d + 1)])

        return polynomial


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):            
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

        
        



if __name__ == "__main__":
    unittest.main()
