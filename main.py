from typing import Callable, Optional

import numpy as np

Vector = np.ndarray
FunctionType = Callable[[np.ndarray], np.ndarray]
Number = int | float | np.number

tol = 1e-12


class fractal2D:
    zeroes: list[Vector] = []

    def __init__(self, f: FunctionType, f_prime: Optional[FunctionType] = None):
        self.f = f
        self.f_prime = f_prime

    def newtons_method(self, guess: Vector) -> Optional[Vector]:
        """Task 2: Performs Newton's method on function `self.f` using `guess` as a starting point."""
        # TODO: handle divergence
        x_n = guess
        while np.linalg.norm(self.f(x_n)) > tol:
            x_n = x_n - np.linalg.inv(self.get_jacobian_matrix()) @ self.f(x_n)
            if np.linalg.norm(x_n) > 100000:  # TODO: make it a reasonable number
                return None

        return x_n

    def zeros_idx(self, guess: Vector) -> Optional[int]:
        """Task 3"""
        # TODO
        new_zero = self.newtons_method(guess)
        if new_zero is None:
            # newton's method did not converge
            return None

        for idx, z in enumerate(self.zeroes):
            if np.abs(new_zero - z) < tol:
                return idx  # index of the zero

        self.zeroes.append(new_zero)
        return len(self.zeroes) - 1  # index of the last zero

    def get_jacobian_matrix(self, guess: Vector) -> np.ndarray:
        """For finding the derivative"""

        h = 0.0001

        guess2 = np.array([guess[0] + h, guess[1]])
        guess3 = np.array([guess[0], guess[1] + h])
        del_f1_x =(self.f(guess2)[0]-self.f(guess)[0]) / h
        del_f1_y = (self.f(guess3)[0]-self.f(guess)[0]) / h
        del_f2_x = (self.f(guess2)[1]-self.f(guess)[1]) / h
        del_f2_y = (self.f(guess3)[1]-self.f(guess)[1]) / h


        pass

    def plot(self, vectors: list[Vector], N: int, coord: tuple[float]) -> None:
        a, b, c, d = coord
        X, Y = np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N))

        # by default everything is -1
        A = np.zeros((N, N)) - 1

        # VERY slow!!! FIXME
        for i_y, y in enumerate(Y):
            for i_x, x in enumerate(X):
                # replace i_y, i_x'th entry with the index of zero or otherwise -2
                A[i_y, i_x] = self.zeros_idx(np.array([y, x])) if not None else -2


def F(x):
    x1 = x[0]
    x2 = x[1]
    return np.array(
        [x1**3 - 3*x1*x2**2 - 1,
         3*x1**2*x2 - x2**3]
    )

def main():
    frac = fractal2D(F)
    frac.get_jacobian_matrix([10,11])


if __name__ == "__main__":
    main()

# Hello :)
