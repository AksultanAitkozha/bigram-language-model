import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
import pandas as pd
import matplotlib.pyplot as plt

bigrams = defaultdict(int)

with open("names.txt", "r") as file:
    for line in file:
        words = line.split()
        for word in words:
            bigram = '^' + word[0]
            bigrams[bigram] += 1
            for i in range(len(word)-1):
                bigram = word[i:i+2]
                bigrams[bigram] += 1
            bigram = word[-1] + '$'
            bigrams[bigram] += 1

total_bigrams = sum(bigrams.values())
probabilities = {bigram: frequency/total_bigrams for bigram, frequency in bigrams.items()}

df = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probability'])
df.index.name = 'Bigram'
styled_df = df.style.background_gradient()
print(styled_df.to_string())

df.plot(kind='bar', legend=None)
plt.ylabel('Probability')
plt.title('Bigram Probabilities')
plt.show()

bigram_probabilities = [(bigram, probability) for bigram, probability in probabilities.items()]
#create a mapping between bigrams and indices to facilitate their use in the neural network
bigram_to_idx = {bigram: idx for idx, bigram in enumerate(probabilities.keys())}

class NameGenerator(nn.Module):
    def __init__(self, bigram_probs, input_size, hidden_size, output_size):
        super(NameGenerator, self).__init__()
        self.bigram_probs = bigram_probs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        x = x.view(1, 1, -1)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

def generate(model):
    name = random.choice([key for key in bigram_to_idx.keys() if key[0] == '^'])[1]

    while name[-1] != '$':
        last_char = name[-1]
        next_bigrams = {bigram: probability for bigram, probability in probabilities.items() if bigram[0] == last_char}

        if not next_bigrams:
            break

        next_bigram_probs = torch.tensor([probability for bigram, probability in next_bigrams.items()], dtype=torch.float32)
        next_bigram_idx = torch.multinomial(next_bigram_probs, 1).item()
        next_bigram = list(next_bigrams.keys())[next_bigram_idx]
        name += next_bigram[1]

        if next_bigram[1] == '$':
            name = name[:-1]
            break
    return name

input_size = len(bigrams)
hidden_size = 128
output_size = len(bigrams)

model = NameGenerator(bigram_probabilities, input_size, hidden_size, output_size)
#the loss function (Negative Log-Likelihood Loss)
criterion = nn.NLLLoss()
#the optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
batch_size = 1

for epoch in range(epochs):
    for i in range(batch_size):
        model.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)
    #a generated name may contain bigrams that are not present in the probabilities dictionary, so
    #we need to update the training loop to ensure that we only generate valid names
    valid_name = False
    while not valid_name:
        try:
            name = generate(model)
            input_tensor = torch.tensor([bigram_to_idx[name[i:i+2]] for i in range(len(name) - 1)], dtype=torch.long)
            target_tensor = torch.tensor([bigram_to_idx[name[i+1:i+3]] for i in range(len(name) - 1)], dtype=torch.long)
            valid_name = True
        except KeyError:
            continue
        hidden = model.init_hidden()
        for input_value, target_value in zip(input_tensor, target_tensor):
            input_value = input_value.unsqueeze(0)
            output, hidden = model(input_value, hidden)
            loss += criterion(output, target_value.unsqueeze(0))

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item() / len(name)}")

#print the name
print("Generated name:")
print(generate(model))
