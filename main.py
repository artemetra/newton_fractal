from typing import Callable, Optional

import numpy as np

Vector = np.ndarray
FunctionType = Callable[[np.ndarray], np.ndarray]
Number = int | float | np.number


class fractal2D:
    zeroes: list[Vector] = []

    def __init__(self, f: FunctionType, f_prime: Optional[FunctionType]):
        self.f = f
        self.f_prime = f_prime

    def newtons_method(self, guess: Vector) -> Vector:
        """Performs Newton's method on function `self.f` using `guess` as a starting point."""
        # TODO
        pass

    def add_to_zeroes(self, guess: Vector):
        """Task 3"""
        # TODO
        pass


def main():
    print("this is a test from Björn")


if __name__ == "__main__":
    main()
