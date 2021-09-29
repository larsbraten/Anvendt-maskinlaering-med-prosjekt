# We're making a model for the NAND operator
import torch
import matplotlib.pyplot as plt
import numpy as np

# Observed/training input and output.
x_train = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)


class NANDModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    # @ sign stands for matrix multiplication
    def f(self, x1, x2):
        return torch.sigmoid((x1 @ self.W[0]) + (x2 @ self.W[1]) + self.b)

    # Logits
    def logits(self, x1, x2):
        return ((x1 @ self.W[0]) + (x2 @ self.W[1]) + self.b).reshape(-1, 1)

    # Uses Cross Entropy
    # Function that measures Binary Cross Entropy between target and output logits.
    def loss(self, x1, x2, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x1, x2), y)


NANDOperatorModel = NANDModel()
epoch = 25000
lr = 0.1
# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([NANDOperatorModel.b, NANDOperatorModel.W, NANDOperatorModel.W], lr)
for i in range(epoch):
    NANDOperatorModel.loss(x_train[:, 0].reshape(-1, 1),
                           x_train[:, 1].reshape(-1, 1),
                           y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

print('W (Weight) = {}, b (bias) = {}, loss value = {}'
      .format(NANDOperatorModel.W.data.numpy()[0], NANDOperatorModel.b.data.numpy()[0],
              (NANDOperatorModel.loss(x_train[:, 0].reshape(-1, 1),
                                      x_train[:, 1].reshape(-1, 1),
                                      y_train))))

# Visualize result
fig = plt.figure('Oppgave B')
plot = fig.add_subplot(111, projection='3d')

# Return coordinate matrices from coordinate vectors.
x1_grid, x2_grid = np.meshgrid(
    np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
y_grid = np.empty([10, 10], dtype=np.float)
# Initialize x-coordinates
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        tenseX = torch.tensor(float(x1_grid[i, j])).reshape(-1, 1)
        tenseY = torch.tensor(float(x2_grid[i, j])).reshape(-1, 1)
# Initialize y-coordinates
        y_grid[i, j] = NANDOperatorModel.f(tenseX, tenseY)
plot_f = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

# Plots coordinates
plot.plot(x_train[:, 0].squeeze(),
          x_train[:, 1].squeeze(),
          y_train[:, 0].squeeze(),
          'o',
          color="red")

plot.set_xlabel("$x_1$")
plot.set_ylabel("$x_2$")
plot.set_zlabel("$y$")
plot.set_xticks([0, 1])
plot.set_yticks([0, 1])
plot.set_zticks([0, 1])
plot.set_xlim(-0.25, 1.25)
plot.set_ylim(-0.25, 1.25)
plot.set_zlim(-0.25, 1.25)

table = plt.table(cellText=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$x_2$", "$f(x)$"],
                  cellLoc="center",
                  loc="lower left")
plt.show()
