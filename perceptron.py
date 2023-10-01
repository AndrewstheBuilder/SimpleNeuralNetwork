import random


class Perceptron:
    def __init__(self, n, HEIGHT):
        self.weights = [random.uniform(-1, 1) for _ in range(n-1)]
        self.weights.append(random.uniform(HEIGHT//4, HEIGHT//2))  # bias term
        self.c = 0.001
        self.stats = []  # List to store weights after each update

    def feedforward(self, inputs):
        total = sum(input_val * weight for input_val,
                    weight in zip(inputs, self.weights))
        return self.activate(total)

    def activate(self, total):
        return 1 if total > 0 else -1

    def train(self, inputs, desired, iteration_num):
        guess = self.feedforward(inputs)
        # self.c = 1 / (1 + iteration_num/100)  # Apply Learning Rate Decay
        error = desired - guess
        for i, input_val in enumerate(inputs):
            self.weights[i] += self.c * error * input_val
        self.stats.append(self.weights.copy())