from typing import Callable, Optional
from datetime import datetime
import pathlib

import numpy as np
import matplotlib.pyplot as plt

Vector = np.ndarray
FunctionType = Callable[[Vector], Vector]
JacobianType = list[list[FunctionType]]
Number = int | float | np.number

TOL_NEWTON = 1e-6
TOL_ZEROES = 1e-6
JAC_STEP_SIZE = 1e-5
MAX_I = 100
MAX_NORM = 100000

GROUP_SIZE = 1000


def evaluate_jacobian(jacobian: JacobianType, val: Vector) -> np.ndarray:
    """Artem Lukin: Evaluates a given Jacobian on a vector

    Args:
        jacobian (JacobianType): the Jacobian to use
        val (Vector): a 2d vector of values to plug into the Jacobian

    Returns:
        np.ndarray: the resulting jacobian
    """
    return np.array(
        [
            [jacobian[0][0](*val), jacobian[0][1](*val)],
            [jacobian[1][0](*val), jacobian[1][1](*val)],
        ]
    )


class fractal2D:
    def __init__(self, f: FunctionType, jacobian_f: Optional[JacobianType] = None):
        """Artem Lukin: Initialize fractal2D.

        Args:
            f (FunctionType): the function to evaluate
            jacobian_f (Optional[JacobianType], optional): Jacobian of the function to use,
            if None it is approximated numerically. Defaults to None.
        """
        self.f = f
        self.zeroes: list[Vector] = []

        # How many times has the newton's method been called
        self.newton_calls = 0
        # This is updated every `GROUP_SIZE` newton's method calls
        self.last_grouped_call = datetime.now()
        # N**2, needed to display progress
        self.N_squared = 0

        # if we are given a jacobian, we evaluate it directly
        if jacobian_f is not None:
            self.jac = lambda val: evaluate_jacobian(jacobian_f, val)
        else:
            # otherwise, we approximate it every time
            self.jac = self.get_jacobian_matrix

    def newtons_method(self, guess: Vector) -> tuple[Optional[Vector], int]:
        """Otto Holmström: Performs Newton's method on `self.f` using `guess` as a starting point.

        Args:
            guess (Vector): starting point, a rough guess for where the zero is located

        Returns:
            tuple[Optional[Vector], int]: returns the zero the method converged to or None
            if it didn't converge, and the number of iterations it took,
            or -1 if it ran out of iterations or it was impossible to continue.
        """
        self.print_and_update_progress()
        x_n = guess
        i = 0
        while np.linalg.norm(f_x := self.f(x_n)) > TOL_NEWTON:
            try:
                x_n = x_n - np.linalg.inv(self.jac(x_n)) @ f_x
            except np.linalg.LinAlgError:  # If self.jac(x_n) is singular
                return None, -1
            i += 1
            if i >= MAX_I:
                return None, -1
            if np.linalg.norm(x_n) > MAX_NORM:
                return None, i

        return x_n, i

    def simplified_newtons_method(self, guess: Vector) -> tuple[Optional[Vector], int]:
        """Björn, Task 5: Performs simplified Newton's method on `self.f` using `guess` as a starting point.

        Args:
            guess (Vector): starting point, a rough guess for where the zero is located

        Returns:
            tuple[Optional[Vector], int]: returns the zero the method converged to or None
            if it didn't converge, and the number of iterations it took,
            or -1 if it ran out of iterations or it was impossible to continue.
        """
        x_n = guess
        self.print_and_update_progress()
        try:
            invjac = np.linalg.inv(self.jac(guess))
        except np.linalg.LinAlgError:  # If self.jac(guess) is singular
            return None, -1
        i = 0
        while np.linalg.norm(f_x := self.f(x_n)) > TOL_NEWTON:
            x_n = x_n - invjac @ f_x
            i += 1
            if np.linalg.norm(x_n) > MAX_NORM:
                return None, -1
            if i >= MAX_I:
                return None, -1
        return x_n, i

    def indices_and_iterations(
        self, points: np.ndarray, simplified: bool
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        indices = []
        iterations = []
        for point in points:
            if simplified:
                new_zero, i = self.simplified_newtons_method(point)

            else:
                new_zero, i = self.newtons_method(point)

            if new_zero is None:
                # Method did not converge
                indices.append(-1)
                iterations.append(i)
                continue

            # Check if the zero is already known
            for idx, z in enumerate(self.zeroes):
                if np.linalg.norm(new_zero - z) < TOL_ZEROES:
                    indices.append(idx)
                    iterations.append(i)
                    break
            else:
                # Add the new zero and get its index
                self.zeroes.append(new_zero)
                indices.append(len(self.zeroes) - 1)
                iterations.append(i)

        return np.array(indices), np.array(iterations)  # index of the last zero

    def get_jacobian_matrix(self, guess: Vector) -> np.ndarray:
        """Theo Melin: A numerical approximation for a Jacobian matrix at a point

        Args:
            guess (Vector): the point at which Jacobian is calculated

        Returns:
            np.ndarray: resulting Jacobian
        """
        h = JAC_STEP_SIZE
        f_guess = self.f(guess)

        guess_x = guess + np.array([h, 0])  # takes the guess and adds h to the x value
        guess_y = guess + np.array([0, h])  # same but for y

        # Partial derivatives:
        del_f1_x = (self.f(guess_x)[0] - f_guess[0]) / h
        del_f1_y = (self.f(guess_y)[0] - f_guess[0]) / h
        del_f2_x = (self.f(guess_x)[1] - f_guess[1]) / h
        del_f2_y = (self.f(guess_y)[1] - f_guess[1]) / h

        return np.array([[del_f1_x, del_f1_y], [del_f2_x, del_f2_y]])

    def plot(
        self,
        N: int,
        coord: tuple[float],
        simplified=False,
        show=True,
        highlight_invalid=False,
    ) -> None:
        """Artem Lukin, Yannick Kapelle: Computes and plots the fractal.

        Args:
            N (int): resolution, i.e. number of points in a side
            coord (tuple[float]): coordinates with respect to which the fractal is rendered
            simplified (bool, optional): use simplified Newton's method instead of regular Newton's method. Defaults to False.
            show (bool, optional): show the image instead of saving it to a file. Defaults to True.
            iter (bool, optional): make an iteration plot instead of a `zeroes`-based Newton fractal. Defaults to False.
            highlight_invalid (bool, optional): highlight points that didn't converge with red. Defaults to False.
        """
        a, b, c, d = coord
        # Creates a grid out of two 1-dimensional arrays representing the indexing
        X, Y = np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N))
        # Used for the printing
        self.N_squared = N**2
        # Creates the plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # We use ravel to make the X, Y from two dimensional to one dimensional arrays.
        # Then column stack to actually assign for each y row an x column.
        points = np.column_stack((X.ravel(), Y.ravel()))

        indices, iterations = self.indices_and_iterations(points, simplified)
        ax1.title.set_text("Zero Plot")
        ax2.title.set_text("Iteration_plot")

        A_indices = indices.reshape((N, N))
        A_iterations = iterations.reshape((N, N))

        # Yannick Kapelle, Artem Lukin
        # matplotlib's default ppi is 72. the minimum figsize
        # is picked to be 6 by 6. If N > 72*6 = 432, we scale the
        # figsize accordingly.
        # All of this fixes the issue where individual pixels
        # were of different aspect ratios.
        if N > 72 * 6:
            fig.set_size_inches(N / 72, N / 72)
        else:
            fig.set_size_inches(6, 6)

        # plt.pcolor(A) # we purposefully don't use pcolor as it's slower than imshow

        # First, plot everything
        ax1.imshow(
            A_indices,
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
            extent=(a, b, c, d),
        )

        ax2.imshow(
            A_iterations,
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
            extent=(a, b, c, d),
        )

        if highlight_invalid:
            # mask of all invalid values
            mask_indices = A_indices == -1
            mask_iterations = A_iterations == -1

            # Then, overlay with red all parts that are invalid
            ax1.imshow(
                np.ma.masked_where(~mask_indices, mask_indices),
                cmap="hsv",
                alpha=1,
                origin="lower",
                interpolation="nearest",
                extent=(a, b, c, d),
            )
            ax2.imshow(
                np.ma.masked_where(~mask_iterations, mask_iterations),
                cmap="hsv",
                alpha=1,
                origin="lower",
                interpolation="nearest",
                extent=(a, b, c, d),
            )

        if show:
            plt.show()
        else:
            filename = datetime.now().strftime("%Y-%m-%d, %H-%M-%S.%f") + ".png"
            # make a directory if it doesn't exist yet
            pathlib.Path.mkdir(pathlib.Path("pics"), exist_ok=True)
            plt.savefig(pathlib.Path("pics/" + filename))

    def print_and_update_progress(self):
        """Artem Lukin: Utility function to print the progress of the render and update it"""
        if self.newton_calls % GROUP_SIZE == 0:
            now = datetime.now()
            stri = "{}/{}    took {} to compute {}".format(
                self.newton_calls,
                self.N_squared,
                now - self.last_grouped_call,
                GROUP_SIZE,
            )
            print(stri, end="\r")
            self.last_grouped_call = now
        self.newton_calls += 1
