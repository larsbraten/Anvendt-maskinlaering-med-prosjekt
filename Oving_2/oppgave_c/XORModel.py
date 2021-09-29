# We're making a model for the XOR operator
import torch
import matplotlib.pyplot as plt
import numpy as np
import random


class XORModel:
    def __init__(self):
        self.W1 = torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
                                [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.W2 = torch.tensor([[random.uniform(-1.0, 1.0)], [random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.b1 = torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]], requires_grad=True)
        self.b2 = torch.tensor([[random.uniform(-1.0, 1.0)]], requires_grad=True)

    # @ sign stands for matrix multiplication
    # Implements the sigmoid function
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Implements the sigmoid function
    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


xor_operator_model = XORModel()

# Observed/training input and output.
x_train = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)
epoch = 100000
lr = 1

# Stochastic gradient descent
optimizer = torch.optim.SGD(
    [xor_operator_model.b1, xor_operator_model.b2, xor_operator_model.W1, xor_operator_model.W2], lr=lr)
for i in range(epoch):
    xor_operator_model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

print(f'W1 = {xor_operator_model.W1.detach().numpy()[0]}, W2 = {xor_operator_model.W2.detach().numpy()[0]}, '
      f'b1 = {xor_operator_model.b1.detach().numpy()[0]}, b2 = {xor_operator_model.b2.detach().numpy()[0]},'
      f' loss = {xor_operator_model.loss(x_train.reshape(-1, 2), y_train)}')

# Add an axes to the current figure and make it the current axes.
ax = plt.axes(projection="3d")

# Sets X, Y and Z limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Sets X, Y and Z labels
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')

# Small table which displays X1, X2 and f(x)
plt.table(cellText=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x_1$", "$x_2$", "$f(x)$"],
          cellLoc="left",
          loc="upper left")
# Inserts X (Z) axis
x1 = np.arange(0, 1, 0.02)
x2 = np.arange(0, 1, 0.02)

# Calculate y coordinates
y = np.empty([len(x1), len(x2)], dtype=np.double)
for t in range(len(x1)):
    for r in range(len(x2)):
        y[t, r] = float(xor_operator_model.f(torch.tensor([float(x1[t]), float(x2[r])])))

# Generates a grid for x1 and x2
x1, x2 = np.meshgrid(x1, x2)

surfacePlot = ax.plot_wireframe(x1, x2, np.array(y))

# Scatter/plot the points for f(x1, x2) in x_train
xplot = [float(x[0]) for x in x_train]
yplot = [float(x[1]) for x in x_train]
ax.scatter(xplot, yplot, y_train)

float(xor_operator_model.f(torch.tensor([1.0, 0.0])))
plt.show()
