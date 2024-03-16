import numpy as np


class LinearRegression:

    def __init__(self, x, y, learning_rate):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate

    def compute_cost(self, x, y, w, b):
        m = x.shape[0]

        y_hat = np.dot(x, w) + b

        cost = ((y_hat - y) ** 2)

        total_cost = (1 / (2 * m)) * sum(cost)

        return total_cost

    def predict(self, x):
        return np.dot(x, self.w) + self.b
