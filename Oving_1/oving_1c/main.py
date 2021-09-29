# Predict head circumference based on age in days
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import timeit

torch.set_printoptions(precision=8)
start = timeit.default_timer()
data = pd.read_csv('day_head_circumference.csv')
# Slicing datasets
x = data.drop(['head circumference'], axis=1).values
y = data['head circumference'].values

# Create tensors and test them by printing
x_train = torch.FloatTensor(x).reshape(-1, 1)
y_train = torch.FloatTensor(y).reshape(-1, 1)


class NonLinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        # Model variables
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    # Predictor, applies the Sigmoid function
    # f (x) = 20σ(xW + b) + 31,
    # σ = Sigmoid function = 1 / (1+e^-z)
    def forward(self, x):
        sigmoid = nn.Sigmoid()
        return 20 * sigmoid(x * self.linear.weight + self.linear.bias) + 31


regression_model = NonLinearRegressionModel(1, 1)
# values for W(weight) and b(bias)
W = regression_model.linear.weight.item()
b = regression_model.linear.bias.item()

# Learning rate, the amount weights are updated during training
lr = 0.0001
# Number of epochs
epochs = 250000
# calculates the mean squared error (squared L2 norm) between each element in the input x and target y.
loss_func = nn.MSELoss()
# Current loss value
current_loss = 0
# Array for storing loss values
loss_list = []

# Using Adam as SGD resulted in a straight line for some reason
optimizer = torch.optim.Adam(regression_model.parameters(), lr=lr)


print("Calculating loss value...")
for i in range(epochs):
    # Finds the next predicate value
    y_predicate = regression_model.forward(x_train)
    # Calculates the mean squared error (squared L2 norm) between each element in the input x and target y.
    current_loss = loss_func(y_predicate, y_train)
    # Stores the calculated mean squared error into an array
    loss_list.append(current_loss)
    # Sets the gradients of all optimized torch.Tensor s to zero.
    optimizer.zero_grad()
    # Computes gradients
    current_loss.backward()
    # Performs a single optimization step
    optimizer.step()

    i += 1
stop = timeit.default_timer()
print(f"Loss value: {current_loss} W: {W} b: {b} "
      f"\nUsing {epochs} epochs, and {lr} as learning rate. "
      f"\nTime spent: {stop - start} seconds")
# Creates a scatter plot containing X and Y values
plt.scatter(x_train, y_train)
# Draws the calculated regression line
plt.plot(x_train, regression_model.forward(x_train).detach(), 'r.')
# Displays the plot
plt.show()
