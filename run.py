from datetime import datetime

import numpy as np

from fractal import fractal2D

def F(x):
    x1 = x[0]
    x2 = x[1]
    # it is (empirically) faster to return a list instead of numpy array
    return [x1**3 - 3 * x1 * x2**2 - 1, 3 * x1**2 * x2 - x2**3]


def F1_Task8(x):
    x1 = x[0]
    x2 = x[1]
    return [x1**3 - 3 * x1 * x2**2 - 2 * x1 - 2, 3 * x1**2 * x2 - x2**3 - 2 * x2]


def F2_Task8(x):
    x1 = x[0]
    x2 = x[1]
    return [
        x1**8
        - 28 * x1**6 * x2**2
        + 70 * x1**4 * x2**4
        + 15 * x1**4
        - 28 * x1**2 * x2**6
        - 90 * x1**2 * x2**2
        + x2**8
        + 15 * x2**4
        - 16,
        8 * x1**7 * x2
        - 56 * x1**5 * x2**3
        + 56 * x1**3 * x2**5
        + 60 * x1**3 * x2
        - 8 * x1 * x2**7
        - 60 * x1 * x2**3,
    ]

def F3_pretty(x):
    x1 = x[0]
    x2 = x[1]

    return [np.cos(x1)-x2**2, x1**3 - 3*x2 - 3]

def F4(x):
    return F(F(F(x)))


def main():
    # the pre-calculated jacobian for function `F`
    jac_F = [  # noqa: F841
        [lambda x, y: 3 * x**2 - 3 * y**2, lambda x, y: -6 * x * y],
        [lambda x, y: 6 * x * y, lambda x, y: 3 * x**2 - 3 * y**2],
    ]
    frac = fractal2D(F3_pretty)
    start = datetime.now()
    print(f"start: {start}")
    frac.plot(N=1000, coord=(-1, 1, -1, 1), simplified=False, show=False, iter=True)
    print(f"duration: {datetime.now()-start}")

if __name__ == "__main__":
    main()