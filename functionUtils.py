import time
import numpy as np


def TIMED(function):
    def wrapper(*args, **kwargs):
        T = time.time()
        result = function(*args, **kwargs)
        T = time.time() - T
        print(">>>function {} timed {}".format(function.__name__, T))
        return result

    return wrapper


def DELAYED(delay):
    def decorate(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            time.sleep(delay)
            return result

        return wrapper

    return decorate


def NOISY(noise):
    def decorate(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            result += np.random.randn() * noise
            return result

        return wrapper

    return decorate


def FLOAT32(function):
    def wrapper(*args, **kwargs):
        args = [np.float32(a) for a in args]
        kwargs = {a: np.float32(kwargs[a]) for a in kwargs}
        result = function(*args, **kwargs)
        result = np.float32(result)
        return result

    return wrapper


def RESTRICT_INVOCATIONS(n):
    def decorate(function):
        function.__n = n

        def wrapper(*args, **kwargs):
            function.__n -= 1
            if function.__n < 0:
                raise EOFError("Exceeded the maximal number of invocations.")
            result = function(*args, **kwargs)
            return result

        return wrapper

    return decorate


def SAVEARGS(**savedkwargs):
    def decorate(function):
        def wrapper(*args, **kwargs):
            result=function(*args,**{**savedkwargs,**kwargs})
            return result
        return wrapper
    return decorate
        

class AbstractShape:
    """
    An abstract class that represents a closed shape.
    """

    def __init__(self):
        raise NotImplementedError()
        pass

    def sample(self):
        """
        Returns
        -------
        A random point (x,y) on the shape contour.

        """
        raise NotImplementedError()

        return (0, 0)

    def contour(self, n: int):
        """
        This method is used to draw the shape contour. It returns an array of
        consecutive points on the shape contour. Larger n results in smoother
        shapes.

        Parameters
        ----------
        n : int
            the number of points on the shape contour to return.

        Returns
        -------
        np.ndarray((n,2))
            An array of consecutive points on the shape contour.

        """
        raise NotImplementedError()

        return np.ndarray((n, 2), dtype=np.float32)

    def area(self) -> np.float32:
        """

        Returns
        -------
        float
            The area of the shape.

        """

        raise NotImplementedError()
        return 1.0


@RESTRICT_INVOCATIONS(1)
@FLOAT32
@TIMED
@NOISY(0.1)
@DELAYED(delay=3)
def plus(a, b):
    return a + b

def minus(a,b):
    return a-b
if __name__ == "__main__":
    min=RESTRICT_INVOCATIONS(1)(minus)
    print(min(3, 1.0 / 11))
    # print(min(3, 1.0 / 11))
    print([f(5) for f in [SAVEARGS(a=a)(lambda x,a: x+a) for a in range(3)]])