import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import timeit

torch.set_printoptions(precision=8)
# Used to measure time expenditure
start = timeit.default_timer()
data = pd.read_csv('day_length_weight.csv')
# Removes day from the CSV
x = data.drop(['day'], axis=1).values
# Removes everything buy day from the CSV
y = data['day'].values

# Creates FloatTensors
x_train = torch.FloatTensor(x).reshape(-1, 2)
y_train = torch.FloatTensor(y).reshape(-1, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Applies a linear transformation to the incoming data
        self.linear = nn.Linear(in_features, out_features)

    # Calculates predicate values
    def forward(self, x):
        return self.linear(x)


linear_regression_model = LinearRegressionModel(2, 1)
lr = 0.0001
epochs = 250000
# calculates the mean squared error (squared L2 norm) between each element in the input x and target y.
loss_func = nn.MSELoss()
# SGD stands for Stochastic Gradient Descent. Optimization algorithm.
optimizer = torch.optim.SGD(linear_regression_model.parameters(), lr=lr)
# Current loss value
current_loss = 0
# Array for storing loss values
loss_list = []
print("Calculating loss value...")
for i in range(epochs):
    # Finds the next predicate value
    y_predicate = linear_regression_model.forward(x_train)
    # Calculates the mean squared error (squared L2 norm) between each element in the input x and target y.
    current_loss = loss_func(y_predicate, y_train)
    # Stores the calculated mean squared error into the back of loss_list array
    loss_list.append(current_loss)
    # Sets the gradients of all optimized torch.Tensor s to zero.
    optimizer.zero_grad()
    # Computes gradients
    current_loss.backward()
    # Performs a single optimization step
    optimizer.step()

    i += 1
# values for W(weight) and b(bias)
W1 = linear_regression_model.linear.weight[0, 0]
W2 = linear_regression_model.linear.weight[0, 1]
b = linear_regression_model.linear.bias.item()
stop = timeit.default_timer()
fig = plt.figure()
plot = fig.add_subplot(111, projection='3d')
# Prints the last recorded loss value
print(f"Loss value: {current_loss} W1: {W1}, W2: {W2}, b: {b}, \nUsing {epochs} epochs, and {lr} as learning rate. "
      f"\nTime spent: {stop - start} seconds")
# Draws test data (points)
plot.scatter(x_train[:, 0], x_train[:, 1], y_train)

# Initializes the x-grids
x1_grid, x2_grid = torch.meshgrid(torch.linspace(1, torch.max(x_train[:, 0] + 1), 10),
                                  torch.linspace(1, torch.max(x_train[:, 1] + 1), 10))

# Initializes the Y-grid
y_grid = torch.empty([10, 10])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        x_plot = torch.FloatTensor([[x1_grid[i, j], x2_grid[i, j]]])
        y_grid[i, j] = linear_regression_model.forward(x_plot)

plot_f = plot.plot_wireframe(x1_grid.detach(), x2_grid.detach(), y_grid.detach(), color='red')
# Displays the plot
plt.show()
