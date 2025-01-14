"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:

    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def unique_roots(self, func, list_of_roots, first_root, maxer):
        unique_list_of_roots = [first_root]
        for root in list_of_roots:
            if not (abs(root - first_root) <= maxer * 2 and abs(
                    func(root) - func(first_root)) <= maxer * 2):
                unique_list_of_roots.append(root)
        return unique_list_of_roots

    def starting_points(self, a, b, f, df):
        current_x = random.uniform(a, b)
        x0 = current_x
        rate = 0.01
        precision = 0.000001
        previous_step_size = 1
        max_iters = 10000
        iters = 0

        if f(x0) > 0:
            while previous_step_size > precision and iters < max_iters:
                prev_x = current_x
                current_x = current_x - rate * df(prev_x)
                if current_x <= a:
                    current_x = a
                    break
                elif current_x >= b:
                    current_x = b
                    break
                if f(current_x) * f(x0) < 0:
                    break

                previous_step_size = abs(current_x - prev_x)
                iters = iters + 1

        else:
            while previous_step_size > precision and iters < max_iters:
                prev_x = current_x
                current_x = current_x + rate * df(prev_x)
                if current_x <= a:
                    current_x = a
                    break
                elif current_x >= b:
                    current_x = b
                    break
                if f(current_x) * f(x0) < 0:
                    break

                previous_step_size = abs(current_x - prev_x)
                iters = iters + 1

        return x0, current_x

    def find_root(self, f_subtract: callable, left_a: float, right_b: float, maxerr=0.001):
        root_found = []
        dt = 0.01
        f_substract_derivative = lambda x: (f_subtract(x + dt) - f_subtract(x - dt)) / (2 * dt)
        x0, x2 = self.starting_points(left_a, right_b, f_subtract, f_substract_derivative)

        if abs(f_subtract(x0)) <= maxerr:
            root_found.append(x0)
            return root_found
        elif abs(f_subtract(x2)) <= maxerr:
            root_found.append(x2)
            return root_found
        if np.sign(f_subtract(x0)) == np.sign(f_subtract(x2)):
            return root_found

        if f_subtract(x0) < 0:
            negative_bound_x = x0
            positive_bound_x = x2
        else:
            negative_bound_x = x2
            positive_bound_x = x0

        for i in range(1000):

            step = (negative_bound_x + positive_bound_x) / 2
            if f_subtract(step) > 0:
                positive_bound_x = step
            else:
                negative_bound_x = step

            for j in range(1000):
                if abs(f_subtract(step)) <= maxerr:
                    root_found.append(step)
                    return root_found
                elif f_substract_derivative(step) != 0:
                    previous_step = step
                    step = step - (f_subtract(step) / f_substract_derivative(step))
                    if (step <= negative_bound_x or step >= positive_bound_x) \
                            or (abs(f_subtract(step) * 2) > (step - previous_step) * f_substract_derivative(step)):
                        break
                    else:
                        if f_subtract(step) > 0:
                            positive_bound_x = step
                        else:
                            negative_bound_x = step
                else:
                    continue


    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        substract_func = lambda x: f1(x) - f2(x)
        ranges_of_finding_root = np.linspace(a, b, num= 90)
        left_bound_iter = 0
        right_bound_iter = 1
        roots_list = []
        while right_bound_iter < len(ranges_of_finding_root):
            root_to_add = self.find_root(substract_func, ranges_of_finding_root[left_bound_iter], ranges_of_finding_root[right_bound_iter], maxerr=maxerr)
            if root_to_add is None:
                left_bound_iter += 1
                right_bound_iter += 1
                continue
            roots_list += root_to_add
            left_bound_iter += 1
            right_bound_iter += 1

        for root2 in roots_list:
            roots_list = self.unique_roots(substract_func, roots_list, root2, maxer=maxerr)
        return roots_list


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()

