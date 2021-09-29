# Not operator model. Same logic as the first Exercise.
import torch
import matplotlib.pyplot as plt

x_train = torch.FloatTensor([[0.0], [1.0]]).reshape(-1, 1)
y_train = torch.FloatTensor([[1.0], [0.0]]).reshape(-1, 1)


class NotOperatorModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    # @ sign stands for matrix multiplication
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    # Logits  ln (p)/(1-p)
    def logits(self, x):
        return x @ self.W + self.b

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


not_operator_model = NotOperatorModel()
epoch = 25000
lr = 0.1
W = not_operator_model.W.data.numpy()[0]
b = not_operator_model.b.data.numpy()[0]
# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([not_operator_model.b, not_operator_model.W], lr)
for i in range(epoch):
    # Compute loss gradients
    not_operator_model.loss(x_train, y_train).backward()
    # Perform optimization by adjusting W and b,
    optimizer.step()
    # Sets the gradients of all optimized torch.Tensors to zero
    optimizer.zero_grad()
print(f"W = {W}, b = {b}, Loss = {not_operator_model.loss(x_train, y_train)}\nUsing {epoch} epochs and {lr} as "
      f"learning rate.")

# Visualize result
plt.title("Oppgave a")
plt.table(cellText=[[0, 1], [1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x$", "$f(x)$"],
          cellLoc="center",
          loc="lower left")
plt.scatter(x_train, y_train)
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0, 1.0, 0.001).reshape(-1, 1)
y = not_operator_model.f(x).detach()
plt.plot(x, y, color="orange")
plt.show()
