"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random

class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    # solving the linear system
    def thomas_algorithm(self, A, b):
        # Extract the size of the matrix and the right-hand side vector
        n = len(b)

        # Make a copy of the original matrix and right-hand side vector
        # so that we can modify them without affecting the original ones
        A = A.copy()
        b = b.copy()

        # Perform the forward sweep
        for i in range(n - 1):
            A[i + 1, i] /= A[i, i]
            A[i + 1, i + 1] -= A[i + 1, i] * A[i, i + 1]
            b[i + 1] -= A[i + 1, i] * b[i]

        # Perform the backward sweep
        x = np.empty(n)
        x[-1] = b[-1] / A[-1, -1]
        for i in range(n - 2, -1, -1):
            x[i] = (b[i] - A[i, i + 1] * x[i + 1]) / A[i, i]
        return np.array([x]).transpose()

    # 4 functions bellow are ment to find good control points (by the presentation)
    def calc_w(self, n):
        D = 4 * np.identity(n)
        np.fill_diagonal(D[1:], 1)
        np.fill_diagonal(D[:, 1:], 1)
        D[0, 0] = 2
        D[n - 1, n - 1] = 7
        D[n - 1, n - 2] = 2
        return D

    def calc_g(self, e, n):
        G = [2 * (2 * e[i] + e[i + 1]) for i in range(n - 1)]
        G[0] = e[0] + 2 * e[1]
        G[n - 2] = 8 * e[n - 2] + e[-1]
        return np.array([G]).transpose()

    def ai(self, e, n):
        W = self.calc_w(n - 1)
        K = self.calc_g(e[0], n)
        return self.thomas_algorithm(W, K)

    def bi(self, e, a, n):
        b = []
        e = e.transpose()
        for i in range(n - 2):
            b.append(2 * e[i + 1][0] - a[i + 1][0])
        b.append((a[n - 2][0] + e[n - 1][0]) / 2)
        b = np.array([b])
        return b.transpose()

    def interpolate(self, f, a, b, n):
        xs = np.linspace(a, b, n)
        ys = list(map(f, xs))
        # Compute the control points
        control_points = []
        ax = self.ai(np.array([xs]), n)
        ay = self.ai(np.array([ys]), n)
        bx = self.bi(np.array([xs]), ax, n).transpose()[0]
        by = self.bi(np.array([ys]), ay, n).transpose()[0]
        ax = ax.transpose()[0]
        ay = ay.transpose()[0]
        for i in range(n - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]

            # Append the control points to the list - by the presentation
            control_points.append((x0, y0))
            control_points.append((ax[i], ay[i]))
            control_points.append((bx[i], by[i]))

        control_points.append((x1, y1))

        def bezier(x):
            # Find the subinterval that x belongs to
            for i in range(n - 1):
                if xs[i] <= x <= xs[i + 1]:
                    interval = i
                    break

            # Evaluate the Bezier curve at the right x -
            # because we add 3 points to each interval we multiplicat by 3 get the correct interval

            p0 = control_points[interval * 3]
            p1 = control_points[interval * 3 + 1]
            p2 = control_points[interval * 3 + 2]
            p3 = control_points[interval * 3 + 3]
            t = (x - xs[interval]) / (xs[interval + 1] - xs[interval])
            # Use the formula for a cubic Bezier curve
            y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]

            return y

        return bezier

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
