from math import e, log
from numpy import sin
from numpy.ma import arctan
from numpy import power as pow, empty, float64

from functionUtils import *
from sampleFunctions import *

def f1(x):
    if type(x) is float64:
        return float64(5)
    if type(x) is int:
        return 5
    if type(x) is float:
        return 5.0
    a = empty(len(x))
    a.fill(5)
    return a


@NOISY(1)
def f1_noise(x):
    if type(x) is float64:
        return float64(5)
    if type(x) is int:
        return 5
    if type(x) is float:
        return 5.0
    a = empty(len(x))
    a.fill(5)
    return a



def f2(x):
    return pow(x, 2) - 3 * x + 2


def f2_nr(x):
    return pow(x, 2) - 3 * x + 2


@NOISY(3)
def f2_noise(x):
    return pow(x, 2) - 3 * x + 2


def f3(x):
    return sin(pow(x, 2))


def f3_nr(x):
    return sin(pow(x, 2))


@NOISY(1)
def f3_noise(x):
    return sin(pow(x, 2))


def f4(x):
    return pow(e, -2 * pow(x, 2))


@NOISY(2)
def f4_noise(x):
    return pow(e, -2 * pow(x, 2))


def f5(x):
    return arctan(x)


def f6(x):
    return sin(x) / x


def f7(x):
    return 1 / log(x)
@NOISY(1)
def f7_noise(x):
    return 1 / log(x)

def f8(x):
    return pow(e, pow(e, x))
@NOISY(1)
def f8_noise(x):
    return pow(e, pow(e, x))
def f9(x):
    return log(log(x))
@NOISY(1)
def f9_noise(x):
    return log(log(x))

def f10(x):
    return sin(log(x))


def f10_nr(x):
    return sin(log(x))


def f11(x):
    return pow(2, 1 / pow(x, 2)) * sin(1 / x)


def f12(x):
    return pow(x, 2) - 5 * x + 2


def f12_nr(x):
    return pow(x, 2) - 5 * x + 2


def f13(x):
    return 5 * pow(x, 2) - 10 * x + 1
@NOISY(1)
def f13_noise(x):
    return 5 * pow(x, 2) - 10 * x + 1

def f13_nr(x):
    return 5 * pow(x, 2) - 10 * x + 1



class Polygon(AbstractShape):
    def __init__(self, knots, noise):
        self._knots = knots
        self._noise = noise
        self._n = len(knots)

    def sample(self):
        i = np.random.randint(self._n)
        t = np.random.random()
        
        x1,y1 = self._knots[i-1]
        x2,y2 = self._knots[i]
        
        x = np.random.random()*(x2-x1)+x1
        x += np.random.randn() * self._noise

        y = np.random.random()*(y2-y1)+y1
        y += np.random.randn() * self._noise
        return x, y

    def contour(self, n: int):
        ppf = n // self._n
        rem = n % self._n        
        points = []
        for i in range(self._n):
            ts = np.linspace(0, 1, num=(ppf+2 if i<rem else ppf+1))
            
            x1,y1 = self._knots[i-1]
            x2,y2 = self._knots[i]
            
            for t in ts[0:-1]:
                x = t*(x2-x1)+x1 
                y = t*(y2-y1)+y1
                xy = np.array((x,y))
                points.append(xy)
        points = np.stack(points, axis=0)
        return points

    def area(self):
        a = 0
        for i in range(self._n):
            x1,y1 = self._knots[i-1]
            x2,y2 = self._knots[i]
            a += 0.5*(x2-x1)*(y1+y2)    
        return a
        

class BezierShape(AbstractShape):
    def __init__(self, knots, control, noise):
        self._knots = knots
        self._control = control
        self._noise = noise
        self._n = len(knots)

        self._fs = [
            bezier3(knots[i - 1], control[2 * i], control[2 * i + 1], knots[i])
            for i in range(self._n)
        ]

    def sample(self):
        i = np.random.randint(self._n)
        t = np.random.random()
        x, y = self._fs[i](t)
        x += np.random.randn() * self._noise
        y += np.random.randn() * self._noise
        return x, y
    
    def contour(self, n: int):
        ppf = n // self._n
        rem = n % self._n        
        points = []
        for i in range(self._n):
            ts = np.linspace(0, 1, num=(ppf+2 if i<rem else ppf+1))
            for t in ts[0:-1]:
                x, y = self._fs[i](t)
                xy = np.array((x,y))
                points.append(xy)
        points = np.stack(points, axis=0)
        return points

    def area(self):
        a = 0
        cntr = self.contour(10000)
        for i in range(10000):
            x1,y1 = cntr[i-1]
            x2,y2 = cntr[i]
            if x1!=x2:
                a += 0.5*(x2-x1)*(y1+y2)    
        return a
        
def noisy_circle(cx, cy, radius, noise) -> AbstractShape:
    return Circle(cx, cy, radius, noise).sample


def shape1()->AbstractShape:
    return Polygon(
            knots=[(0,0),(1,1),(1,-1)],
            noise=0.1
            )

def shape2()->AbstractShape:
    return Polygon(
            knots=[(0,0),(0,1),(1,1),(1,0)],
            noise=0.1
            )

def shape3()->AbstractShape:
    return Polygon(
            knots=[(0,0),(0,8),(20,1),(1,0)],
            noise=0.3
            )

def shape4()->AbstractShape:
    return Polygon(
            knots=[(0,0),(0.5,100000),(50000,70000),(100000,1),(50000,0)],
            noise=2000
            )

def shape5()->AbstractShape:
    return Polygon(
            knots=[(0,0),(0.5,100000),(1,1),(100000,1),(50000,0)],
            noise=1.0
            )

def shape6()->AbstractShape:
    return BezierShape(
                knots=[[0, 0], [0, 1]],
                control = [[2, 0.5], [2, 0.5], [-2, 0.5], [-2, 0.5]],
                noise=0.1
                )

def shape7()->AbstractShape:
    return BezierShape(
            knots=[(0,0),(1,1),(1,-1)],
            control=[(0.2,8),(0.8,6),(3,0.5),(-1,0.5),(3,-5),(0.2,-0.6)],
            noise=0.1
            )



if __name__ == '__main__':
    r1 = f1(3)
    r2 = f2(3)
    r3 = f3(3)
    r4 = f4(4)
