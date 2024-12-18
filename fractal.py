from typing import Callable, Optional
from datetime import datetime
import pathlib

import numpy as np
import matplotlib.pyplot as plt

Vector = np.ndarray
FunctionType = Callable[[Vector], Vector]
JacobianType = list[list[FunctionType]]
Number = int | float | np.number

TOL_NEWTON = 1e-5
TOL_ZEROES = 1e-5
JAC_STEP_SIZE = 1e-5
MAX_I = 50
MAX_NORM = 100000

GROUP_SIZE = 1000


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
        #Artem Lukin
        self.f = f
        self.newton_calls = 0
        self.last_grouped_call = datetime.now()
        self.N_squared = 0
        # if we are given a jacobian, we evaluate it directly
        if jacobian_f is not None:
            self.jac = lambda val: evaluate_jacobian(jacobian_f, val)
        else:
            # otherwise, we approximate it every time
            self.jac = self.get_jacobian_matrix

    def newtons_method(self, guess: Vector) -> tuple[Optional[Vector], int]:
        # Otto Holmström
        """Task 2: Performs regular Newton's method on function
        `self.f` using `guess` as a starting point.

        Returns the tuple of the zero found or None if the algorithm didn't converge,
        and the last iteration count, or -1 if the method ran out of iterations
        """
        self.print_and_update_progress()
        x_n = guess
        i = 0
        while np.linalg.norm(Reused_Calc :=self.f(x_n)) > TOL_NEWTON:
            try:
                x_n = x_n - np.linalg.inv(self.jac(x_n)) @ Reused_Calc
            except np.linalg.LinAlgError:  # If self.jac(x_n) is singular
                return None, -1
            i += 1
            if i >= MAX_I:
                return None, -1
            if np.linalg.norm(x_n) > MAX_NORM:
                return None, i

        return x_n, i
    
    def simplified_newtons_method(self, guess: Vector) -> tuple[Optional[Vector], int]:
        #Björn
        """Task 5: Performs simplified Newton's method on function
        `self.f` using `guess` as a starting point.

        Returns the tuple of the zero found or None if the algorithm didn't converge,
        and the last iteration count, or -1 if the method ran out of iterations
        """
        x_n = guess
        self.print_and_update_progress()
        try:
            invjac = np.linalg.inv(self.jac(guess))
        except np.linalg.LinAlgError:  # If self.jac(guess) is singular
            return None, -1
        i = 0
        while np.linalg.norm(self.f(x_n)) > TOL_NEWTON:
            x_n = x_n - invjac @ self.f(x_n)
            i += 1
            if np.linalg.norm(x_n) > MAX_NORM:
                return None, i
            if i >= MAX_I:
                return None, -1
        return x_n, i

    def zeros_idx(self, guess: Vector, simplified: bool) -> Optional[int]:
        #Björn
        """Task 3"""
        if simplified:
            new_zero, _ = self.simplified_newtons_method(guess)
        else:
            new_zero, _ = self.newtons_method(guess)

        if new_zero is None:
            # newton's method did not converge
            return None

        for idx, z in enumerate(self.zeroes):
            if np.linalg.norm(new_zero - z) < TOL_ZEROES:
                return idx  # index of the zero

        self.zeroes.append(new_zero)
        return len(self.zeroes) - 1  # index of the last zero

    def get_jacobian_matrix(self, guess: Vector) -> np.ndarray:
        #Theo Melin
        """An approximation for finding the derivative"""

        h = JAC_STEP_SIZE

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
        #Yannick Kapelle
        """Vectorized computation for all points."""  
        indices = []
        for point in points:
            idx = self.zeros_idx(point, simplified)
            indices.append(idx if idx is not None else -2)
        return np.array(indices)

    def compute_iterations(self, points: np.ndarray, simplified: bool) -> np.ndarray:
        """Task 7; Yannick Kapelle"""
        iterations = []
        for point in points:
            if simplified:
                _, iter = self.simplified_newtons_method(point)
            else:
                _, iter = self.newtons_method(point)
            iterations.append(iter)
        return np.array(iterations)

    #Artem Lukin, Yannick Kapelle
    def plot(
        self,
        N: int,
        coord: tuple[float],
        simplified=False,
        show=True,
        iter=False,
        highlight_invalid=False,
    ) -> None:
        # TODO: add documentation
        a, b, c, d = coord
        X, Y = np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N))
        self.N_squared = N**2
        fig, ax = plt.subplots()
        points = np.column_stack((X.ravel(), Y.ravel()))
        # Yannick Kapelle
        if iter:
            result = self.compute_iterations(points, simplified)
            plt.title("Fractal Iterations")
        else:
            result = self.compute_indices(points, simplified)
            plt.title("Newton Fractal")

        A = result.reshape((N, N))

        #Yannick Kapelle
        # matplotlib's default ppi is 72. the minimum figsize
        # is picked to be 6 by 6. If N > 72*6 = 432, we scale the
        # figsize accordingly.
        # All of this fixes the issue where individual pixels
        # were of different aspect ratios.
        if N > 72 * 6:
            fig.set_size_inches(N / 72, N / 72)
        else:
            fig.set_size_inches(6, 6)

        # First, plot everything
        plt.imshow(
            A,
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
            extent=(a, b, c, d),
        )
        # Artem Lukin
        if highlight_invalid:
            # mask of all invalid values
            mask = A == -1
            # Then, overlay with red all parts that are invalid
            plt.imshow(
                np.ma.masked_where(~mask, mask),
                cmap="hsv",
                alpha=1,
                origin="lower",
                interpolation="nearest",
                extent=(a, b, c, d),
            )

        # plt.pcolor(A) # we purposefully don't use pcolor as it's slower than imshow
        if show:
            plt.show()
        else:
            filename = datetime.now().strftime("%Y-%m-%d, %H-%M-%S") + ".png"
            plt.savefig(pathlib.Path("pics/" + filename))
    
    # Artem Lukin
    def print_and_update_progress(self):
        if self.newton_calls % GROUP_SIZE == 0:
            now = datetime.now()
            print(
                f"{self.newton_calls}/{self.N_squared}",
                "   ",
                "took",
                now - self.last_grouped_call,
                "to compute",
                GROUP_SIZE,
                end="\r",
            )
            self.last_grouped_call = now
        self.newton_calls += 1
