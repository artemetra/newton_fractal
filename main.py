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

    def add_to_zeroes(self, guess: Vector) -> Optional[int]:
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

    def get_jacobian_matrix(self, x_n: Vector) -> np.ndarray:
        """For finding the derivative"""
        jac = ...
        self.f([])
        pass

    def plot(
        self, vectors: list[Vector], N: int, coord: tuple[float]
    ) -> None:
        a,b,c,d = coord
        X, Y = np.meshgrid(np.linspace(a,b, N), np.linspace(c,d, N))
        # convert all Xs and Ys into a single array of all points
        points = np.column_stack((X.ravel(), Y.ravel()))
        print(points)

def main():
    f = fractal2D(lambda x: x**2)
    f.plot([np.array((1,2))], N=5, coord=(1,2,3,4))


if __name__ == "__main__":
    main()

# Hello :)
