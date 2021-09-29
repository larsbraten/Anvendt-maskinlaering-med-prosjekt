import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import timeit
torch.set_printoptions(precision=8)
data = pd.read_csv(r'length_weight.csv')
x_train = torch.FloatTensor(data.length).reshape(-1, 1)
y_train = torch.FloatTensor(data.weight).reshape(-1, 1)

# Used to measure time expenditure
start = timeit.default_timer()


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


linear_regression_model = LinearRegressionModel()
epoch = 100000
lr = 0.00015
W = linear_regression_model.W.data.numpy()[0]
b = linear_regression_model.b.data.numpy()[0]

# Optimize: adjust W and b to minimize loss using stochastic gradient descent (Adam)
optimizer = torch.optim.Adam([linear_regression_model.W, linear_regression_model.b], lr)
print("Calculating loss value...")
for i in range(epoch):
    # Compute loss gradients
    linear_regression_model.loss(x_train, y_train).backward()
    # Perform optimization by adjusting W and b,
    optimizer.step()
    # Sets the gradients of all optimized torch.Tensors to zero
    optimizer.zero_grad()
stop = timeit.default_timer()
# Print model variables and loss value.
print(f"W = {W}, b = {b}, Loss = {linear_regression_model.loss(x_train, y_train)}\nUsing {epoch} epochs and {lr} as "
      f"learning rate. \nTime spent: {stop - start}")
# Plot variables
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
# Returns the lowest and the highest values from the x_train set (45. , 110.)
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
# Draws the calculated regression line
plt.plot(x, linear_regression_model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
# Adds previously specified labels to the plot
plt.legend()
plt.show()
