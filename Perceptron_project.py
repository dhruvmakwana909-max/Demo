import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Perceptron class
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.lr = learning_rate

    def predict_one(self, x):
        z = np.dot(self.weights, x) + self.bias
        return step(z)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                output = self.predict_one(xi)
                error = target - output
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def predict(self, X):
        return np.array([self.predict_one(xi) for xi in X])

# Input combinations (truth table inputs)
X = np.array([[0,0],[0,1],[1,0],[1,1]])

# Truth tables for AND, OR, NOR
truth_tables = {
    "AND": np.array([0,0,0,1]),
    "OR":  np.array([0,1,1,1]),
    "NOR": np.array([1,0,0,0])
}

# Setup figure
plt.figure(figsize=(15,5))

# Train and plot each gate
for i, (gate_name, y) in enumerate(truth_tables.items()):
    p = Perceptron(n_inputs=2, learning_rate=0.1)
    p.train(X, y, epochs=10)

    # Predictions
    y_pred = p.predict(X)
    print(f"\n{gate_name} Gate:")
    print("Weights:", p.weights)
    print("Bias:", p.bias)
    print("Predictions:", y_pred)

    # Plot subplot
    plt.subplot(1, 3, i+1)
    for j, point in enumerate(X):
        if y[j] == 0:
            plt.scatter(point[0], point[1], color='red', marker='o', label='Class 0' if j==0 else "")
        else:
            plt.scatter(point[0], point[1], color='blue', marker='x', s=100, label='Class 1' if j==3 else "")

    # Decision boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2
    x_vals = np.linspace(-0.5, 1.5, 100)
    if p.weights[1] != 0:
        y_vals = -(p.weights[0]*x_vals + p.bias) / p.weights[1]
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

    # Graph formatting
    plt.title(f"{gate_name} Gate")
    plt.xlabel("Input 1 (X1)")
    plt.ylabel("Input 2 (X2)")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid(True)
    plt.legend()

plt.suptitle("Perceptron Decision Boundaries for AND, OR, and NOR Gates", fontsize=14)
plt.tight_layout()
plt.show()
