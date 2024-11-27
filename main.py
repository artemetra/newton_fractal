from typing import Callable, Optional

import numpy as np

Vector = np.ndarray
FunctionType = Callable[[np.ndarray], np.ndarray]
Number = int | float | np.number

tol = 1e-12

class fractal2D:
    zeroes: list[Vector] = []

    def __init__(self, f: FunctionType, f_prime: Optional[FunctionType]):
        self.f = f
        self.f_prime = f_prime

    def newtons_method(self, guess: Vector) -> Vector:
        """Task 2: Performs Newton's method on function `self.f` using `guess` as a starting point."""
        # TODO
        
        x_n = guess
        while np.linalg.norm(self.f(x_n)) > tol:
            x_n = x_n - np.linalg.inv(self.get_jacobian_matrix()) @ self.f(x_n)

        return x_n

    def add_to_zeroes(self, guess: Vector):
        """Task 3"""
        # TODO
        pass


    def get_jacobian_matrix(self, x_n: Vector) -> np.ndarray:
        """For finding the derivative"""
        self.f([])
        pass
    


def main():
    print("this is a test from Otto")


if __name__ == "__main__":
    main()
