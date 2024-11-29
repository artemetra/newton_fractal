from typing import Callable, Optional
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


Vector = np.ndarray
FunctionType = Callable[[np.ndarray], np.ndarray]
JacobianType = list[list[FunctionType]]
Number = int | float | np.number

tol = 1e-6

MAX_I = 75
MAX_NORM = 250


def evaluate_jacobian(jacobian: JacobianType, val: Vector) -> np.ndarray:
    return np.array(
        [
            [jacobian[0][0](*val), jacobian[0][1](*val)],
            [jacobian[1][0](*val), jacobian[1][1](*val)],
        ]
    )


class fractal2D:
    zeroes: list[Vector] = []

    def __init__(self, f: FunctionType, jacobian_f: Optional[JacobianType] = None):
        self.f = f

        # if we are given a jacobian, we evaluate it directly
        if jacobian_f is not None:
            self.jac = lambda val: evaluate_jacobian(jacobian_f, val)
            return

        # otherwise, we approximate it every time
        self.jac = lambda val: self.get_jacobian_matrix(val)

    def newtons_method(self, guess: Vector) -> tuple[Optional[Vector], int]:
        """Task 2: Performs regular Newton's method on function
        `self.f` using `guess` as a starting point.

        Returns the tuple of the zero found or None if the algorithm didn't converge,
        and the last iteration count, or -1 if the method ran out of iterations
        """
        x_n = guess
        i = 0
        while np.linalg.norm(self.f(x_n)) > tol:
            x_n = x_n - np.linalg.inv(self.jac(x_n)) @ self.f(x_n)
            i += 1
            if (norm:=np.linalg.norm(x_n)) > MAX_NORM:
                print(f"hit {norm}")
                return None, i
            if i >= MAX_I:
                return None, -1

        return x_n, i

    def zeros_idx(self, guess: Vector, simplified: bool) -> Optional[int]:
        """Task 3"""
        if simplified:
            new_zero, _ = self.simplified_newtons_method(guess)
        else:
            new_zero, _ = self.newtons_method(guess)

        if new_zero is None:
            # newton's method did not converge
            return None

        for idx, z in enumerate(self.zeroes):
            if np.linalg.norm(new_zero - z) < tol:
                return idx  # index of the zero

        self.zeroes.append(new_zero)
        return len(self.zeroes) - 1  # index of the last zero

    def get_jacobian_matrix(self, guess: Vector) -> np.ndarray:
        """An approximation for finding the derivative"""

        h = 0.00001

        guess_x = np.array(
            [guess[0] + h, guess[1]]
        )  # takes the guess and adds h to the x value
        guess_y = np.array([guess[0], guess[1] + h])  # same but for y

        # Partial derivatives:
        del_f1_x = (self.f(guess_x)[0] - self.f(guess)[0]) / h
        del_f1_y = (self.f(guess_y)[0] - self.f(guess)[0]) / h
        del_f2_x = (self.f(guess_x)[1] - self.f(guess)[1]) / h
        del_f2_y = (self.f(guess_y)[1] - self.f(guess)[1]) / h

        return np.array([[del_f1_x, del_f1_y], [del_f2_x, del_f2_y]])

    def compute_indices(self, points: np.ndarray, simplified: bool) -> np.ndarray:
        """Vectorized computation for all points."""  # FIXME: really?
        indices = []
        for point in points:
            idx = self.zeros_idx(point, simplified)
            indices.append(idx if idx is not None else -2)
        return np.array(indices)

    def compute_iterations(self, points: np.ndarray, simplified: bool) -> np.ndarray:
        """Task 7"""
        iterations = []
        for point in points:
            if simplified:
                _, iter = self.simplified_newtons_method(point)
            else:
                _, iter = self.newtons_method(point)
            iterations.append(iter)
        return np.array(iterations)

    def plot(self, N: int, coord: tuple[float], simplified=False) -> None:
        a, b, c, d = coord
        X, Y = np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N))

        # by default everything is -1
        A = np.zeros((N, N)) - 1

        points = np.column_stack((X.ravel(), Y.ravel()))
        # TODO: this is still reeeaaaallllyyy slow
        indices = self.compute_indices(points, simplified)
        A = indices.reshape((N, N))
        plt.pcolor(A)
        # plt.legend()
        plt.show()
        # plt.savefig(f"pics/{datetime.now()}.png")

    def simplified_newtons_method(self, guess: Vector) -> tuple[Optional[Vector], int]:
        """Task 5: Performs simplified Newton's method on function
        `self.f` using `guess` as a starting point.

        Returns the tuple of the zero found or None if the algorithm didn't converge,
        and the last iteration count, or -1 if the method ran out of iterations
        """
        x_n = guess
        invjac = np.linalg.inv(self.jac(guess))
        i = 0
        while np.linalg.norm(self.f(x_n)) > tol:
            x_n = x_n - invjac @ self.f(x_n)
            i += 1
            if np.linalg.norm(x_n) > MAX_NORM:
                return None, i
            if i >= MAX_I:
                return None, -1

        return x_n, i

    def iter_plot(self, N: int, coord: tuple[float], simplified=False) -> None:
        """Task 7"""
        a, b, c, d = coord
        X, Y = np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N))

        # by default everything is -1
        A = np.zeros((N, N)) - 1

        points = np.column_stack((X.ravel(), Y.ravel()))
        indices = self.compute_iterations(points, simplified)
        A = indices.reshape((N, N))
        plt.pcolor(A)
        # plt.legend()
        plt.show()
        # plt.savefig(f"pics/{datetime.now()}.png")


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


def main():
    # the pre-calculated jacobian for function `F`
    jac_F = [  # noqa: F841
        [lambda x, y: 3 * x**2 - 3 * y**2, lambda x, y: -6 * x * y],
        [lambda x, y: 6 * x * y, lambda x, y: 3 * x**2 - 3 * y**2],
    ]
    frac = fractal2D(F1_Task8)
    frac.iter_plot(N=400, coord=(-2, 2, -2, 2), simplified=False)


if __name__ == "__main__":
    main()

# Hello :)
