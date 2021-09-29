# Taken from https://gitlab.com/ntnu-tdat3025/cnn/mnist/-/blob/master/nn.py
# Added dropout
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST(
    './data', train=True, download=True)
# torch.functional.nn.conv2d argument must include channels (1)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()
# Create output tensor
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]),
        mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST(
    './data', train=False, download=True)
# torch.functional.nn.conv2d argument must include channels (1)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]),
       mnist_test.targets] = 1  # Populate output

# Normalization of inputs
# Finds the mean of the training set
mean = x_train.mean()
# Standard deviation
std = x_train.std()

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()
        # Applies a 2D max pooling over an input signal composed of several input planes.
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # First convolution with 1 in channel and 32 out channels
        self.convolution_first = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # Second convolution with 32 in channels and 64 out channels
        self.convolution_second = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Applies a linear transformation to the incoming data
        self.dense1 = nn.Linear(64 * 7 * 7, 1024)
        self.dense2 = nn.Linear(1024, 100)
        self.dense3 = nn.Linear(100, 10)

    def logits(self, x):
        # Dropout randomly zeroes some of the elements of the input tensor
        # with probability p using samples from a Bernoulli distribution.
        # Helps prevent overfitting or "overtraining".
        # Prevents the model from fitting exactly to its training data by stopping it from memorizing noise
        x = self.max_pool(F.dropout((self.convolution_first(x)), p=0.1))
        x = self.max_pool(F.dropout((self.convolution_second(x)), p=0.1))
        # Applies a linear transformation to the incoming data
        # View shares underlying data with its base tensor, which means that if it is edited,
        # the original tensor will also be edited
        x = self.dense1(x.view(-1, 64 * 7 * 7))
        x = self.dense2(x.view(-1, 1024))
        x = self.dense3(x.view(-1, 100))
        return x

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


convolutional_neural_network_model = ConvolutionalNeuralNetworkModel()

epoch = 20
lr = 0.001

# Optimize: adjust W and b to minimize loss using Adam
optimizer = torch.optim.Adam(convolutional_neural_network_model.parameters(), lr=lr)
for i, epoch in enumerate(range(epoch)):
    for batch in range(len(x_train_batches)):
        # Compute loss gradients
        convolutional_neural_network_model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        # Perform optimization by adjusting W and b
        optimizer.step()
        # Clear gradients for next step
        optimizer.zero_grad()

    print("accuracy = %s" % convolutional_neural_network_model.accuracy(x_test, y_test))
