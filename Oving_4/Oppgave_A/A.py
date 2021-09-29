import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    # Reset states prior to new input sequence
    def reset(self):
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, 1, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    # x shape: (sequence length, batch size, encoding size)
    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    # x shape: (sequence length, batch size, encoding size)
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
    def loss(self, x,
             y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0.],  # 'e'
    [0., 0., 0., 1., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 1., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 1., 0., 0.],  # 'w'
    [0., 0., 0., 0., 0., 0., 1., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 1.],  # 'd'
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

x_train = torch.tensor(
    [[char_encodings[0]], [char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]],
     [char_encodings[4]], [char_encodings[0]], [char_encodings[5]], [char_encodings[4]], [char_encodings[6]],
     [char_encodings[3]], [char_encodings[7]], [char_encodings[0]]])  # ' hello world'
y_train = torch.tensor(
    [char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0],
     char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3], char_encodings[7],
     char_encodings[0], char_encodings[1]])  # 'hello world h' Added empty space after world to make batch sizes the same

model = LongShortTermMemoryModel(encoding_size)

# takes the square root of the gradient average before adding epsilon
lr = 0.001
epochs = 500
optimizer = torch.optim.RMSprop(model.parameters(), lr)
for i in range(epochs):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 10 == 9:
        # Generate characters from the initial characters ' h'
        model.reset()
        text = ' h'
        model.f(torch.tensor([[char_encodings[0]]]))
        y = model.f(torch.tensor([[char_encodings[1]]]))
        text += index_to_char[y.argmax(1)]
        for c in range(50):
            y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
            text += index_to_char[y.argmax(1)]
        print(text)
