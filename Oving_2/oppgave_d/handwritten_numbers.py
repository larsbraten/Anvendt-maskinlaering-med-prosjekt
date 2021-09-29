import torch
import torchvision
import matplotlib.pyplot as plt

# Loading of MNIST taken from https://gitlab.com/ntnu-tdat3025/ann/mnist/-/blob/master/main.py
mnist_train = torchvision.datasets.MNIST('', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
# Create output tensor
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]),
        mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()
# Create output tensor
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]),
       mnist_test.targets] = 1  # Populate output


class MNISTRecognitionModel:
    def __init__(self):
        # Returns a tensor filled with the scalar value 1
        self.W = torch.ones([784, 10], requires_grad=True)
        self.b = torch.ones([1, 10], requires_grad=True)

    # Predictor
    # Applies the Softmax function as requested
    def f(self, x):
        return torch.nn.functional.softmax(x @ self.W + self.b, dim=1)

    # Measures Binary Cross Entropy between target and output logits.
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


mnist_recognition_model = MNISTRecognitionModel()
# Learning rate
lr = 0.1
# Epochs
epoch = 2000
# Using Stochastic Gradient Descent
optimizer = torch.optim.SGD([mnist_recognition_model.W, mnist_recognition_model.b], lr=lr)
# List for storing results
resultList = []
# Loss method, declared here to make the loop prettier
loss = mnist_recognition_model.loss
# Accuracy method, declared here to make the loop prettier
accuracy = mnist_recognition_model.accuracy
# Using enumerate to avoid unpacking non-iterable object
for index, epoch in enumerate(range(epoch)):
    # Prints the results of every 100 iteration
    if (index+1) % 100 == 0:
        print(
            f"Epoch = {index + 1}, "
            f"Loss = {loss(x_train, y_train).item()}, "
            f"Accuracy = {accuracy(x_test, y_test).item()}")
        resultList.append([index + 1, loss(x_train, y_train).item(),
                           accuracy(x_test, y_test).item()])
    # Compute loss gradients
    loss(x_train, y_train).backward()
    # Perform optimization by adjusting W and b
    optimizer.step()
    # Sets gradients to none
    optimizer.zero_grad()

fig = plt.figure()
# Superplot title
fig.suptitle('Accuracy: ' + str(accuracy(x_test, y_test).item()))
# Plotting 10 numbers
for i in range(10):
    # Add an Axes to the current figure or retrieve an existing Axes.
    plt.subplot(2, 5, i + 1)
    # Displays the data as an image
    plt.imshow(mnist_recognition_model.W[:, i].detach().numpy().reshape(28, 28))
    # Plot title
    plt.title(f'{i}')
    # sets the current tick locations and labels of the x-axis.
    plt.xticks([])
    # sets the current tick locations and labels of the y-axis.
    plt.yticks([])
# Displays the generated plot
plt.show()
