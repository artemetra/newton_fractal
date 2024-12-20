from datetime import datetime

import numpy as np

from fractal import fractal2D


def F_Task4(x):
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


# === CUSTOM FUNCTIONS ===
def F3_pretty(x):
    x1 = x[0]
    x2 = x[1]

    return [np.cos(x1) - x2**2, x1**3 - 3 * x2 - 3]


def F4(x):
    return F_Task4(F_Task4(F_Task4(x)))


def F5(x):
    x1 = x[0]
    x2 = x[1]
    return [np.cos(x2) ** 2 + x1**3.0, x2**2 + x1**2 - 4 * np.sin(x1 * x2)]


def F6(x):
    x1 = x[0]
    x2 = x[1]
    return [np.acosh(np.abs(x1) + 4) + x2, -(x1**3) + 20 * x2**2]


def F7(x, a):
    x1 = x[0]
    x2 = x[1]
    return [x1**3 - np.cos(a * x1) * x1 * x2**2 - 1, 3 * x1**2 * x2 - a * x2**3]


def F8(x, a):
    x1 = x[0]
    x2 = x[1]
    return [
        x1**8
        - 28 * x1**6 * x2**2
        + 7 * a * x1**4 * x2**4
        + 15 * x1**4
        - 28 * x1**2 * x2**6
        - 90 * x1**2 * x2**2
        + x2**8
        + 15 * x2**4
        - a**2,
        a * x1**7 * x2
        - 56 * x1**5 * x2**3
        + 56 * x1**3 * x2**5
        + 60 * x1**3 * x2
        - a * x1 * x2**7
        - 60 * x1 * x2**3,
    ]


def main():
    # the pre-calculated jacobian for function `F_Task4`
    jac_F = [  # noqa: F841
        [lambda x, y: 3 * x**2 - 3 * y**2, lambda x, y: -6 * x * y],
        [lambda x, y: 6 * x * y, lambda x, y: 3 * x**2 - 3 * y**2],
    ]

    frac = fractal2D(F_Task4)
    start = datetime.now()
    print("start:", start)
    N = 300
    print("N^2:", N**2)
    b = 1
    frac.plot(
        N=N, coord=(-b, b, -b, b), simplified=False, show=False, highlight_invalid=False
    )
    print("\nduration:", datetime.now() - start)


if __name__ == "__main__":
    main()
