import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128)
        self.linear = nn.Linear(128, output_size)

    def reset(self, batch_size):
        zero_state = torch.zeros(1, batch_size, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.linear(out[-1].reshape(-1, 128))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


character_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' 0
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 1
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 2
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'f' 3
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 4
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'l' 5
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'm' 6
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'n' 7
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'o' 8
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p' 9
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'r' 10
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 's' 11
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]   # 't'  12
]

encoding_size = len(character_encodings)

index_to_char = [' ', 'a', 'c', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't']
emoji_encodings = [
    [1., 0., 0., 0., 0., 0., 0.],  # '🎩' 0
    [0., 1., 0., 0., 0., 0., 0.],  # '🐀' 1
    [0., 0., 1., 0., 0., 0., 0.],  # '🐈' 2
    [0., 0., 0., 1., 0., 0., 0.],  # '🏢' 3
    [0., 0., 0., 0., 1., 0., 0.],  # '🧔' 4
    [0., 0., 0., 0., 0., 1., 0.],  # '🧢' 5
    [0., 0., 0., 0., 0., 0., 1.]   # '👦' 6
]

emoji_size = len(emoji_encodings)
index_to_emoji = ['🎩', '🐀', '🐈', '🏢', '🧔', '🧢', '👦']


def encode(string):
    encoding = []

    for char in string:
        encoding.append(character_encodings[index_to_char.index(char)])

    return encoding


def encode_emoji(emoji):
    return emoji_encodings[index_to_emoji.index(emoji)]


def decode_emoji(tensor):
    return index_to_emoji[tensor.argmax(1)]


x_train = torch.tensor([
    encode('hat '),
    encode('rat '),
    encode('cat '),
    encode('flat'),
    encode('matt'),
    encode('cap '),
    encode('son '),
]).transpose(1, 0)

y_train = torch.tensor([encode_emoji('🎩'), encode_emoji('🐀'), encode_emoji('🐈'), encode_emoji('🏢'),
                        encode_emoji('🧔'), encode_emoji('🧢'), encode_emoji('👦')])

model = LSTM(encoding_size, emoji_size)
lr = 0.001
# Takes the square root of the gradient average before adding epsilon
optimizer = torch.optim.RMSprop(model.parameters(), lr)

epochs = 500
for i in range(epochs):
    model.reset(x_train.size(1))
    # Applies cross entropy loss and computes the gradient of current tensor w.r.t. graph leaves.
    model.loss(x_train, y_train).backward()
    # Performs a single optimization step
    optimizer.step()
    optimizer.zero_grad()

    if i % 10 == 9:
        model.reset(1)
        # Prompts a user input
        test_string = input("Enter test string\n")
        print("Did you mean: " + decode_emoji(
            model.f(torch.tensor([encode(test_string)]).transpose(1, 0))) + "?")
      #  test_string1 = "rt "
       # test_string2 = "rats"
        #print("Emoji from gotten from 'rt ' & 'rats'" + decode_emoji(
         #   model.f(torch.tensor([encode(test_string1)]).transpose(1, 0))))
