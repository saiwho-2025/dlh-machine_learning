#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Represents a single neuron for binary classification."""

    def __init__(self, nx):
        """Initialize a neuron with nx input features."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weights vector."""
        return self.__W

    @property
    def b(self):
        """Return the bias."""
        return self.__b

    @property
    def A(self):
        """Return the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate forward propagation."""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculate the logistic regression cost."""
        m = Y.shape[1]

        return -(1 / m) * np.sum(
            Y * np.log(A)
            + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions."""
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent."""
        m = Y.shape[1]

        error = A - Y
        dW = np.matmul(error, X.T) / m
        db = np.sum(error) / m

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")

            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and <= iterations"
                )

        recorded_iterations = []
        recorded_costs = []

        for iteration in range(iterations + 1):
            A = self.forward_prop(X)

            should_record = (
                (verbose or graph)
                and (
                    iteration % step == 0
                    or iteration == iterations
                )
            )

            if should_record:
                current_cost = self.cost(Y, A)

                if verbose:
                    print(
                        "Cost after {} iterations: {}"
                        .format(iteration, current_cost)
                    )

                if graph:
                    recorded_iterations.append(iteration)
                    recorded_costs.append(current_cost)

            if iteration < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(recorded_iterations, recorded_costs, color="blue")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
