import torch
import torch.nn as nn
import re

token_map = {}
counter = 0

parsed_dataset = open("parsed_datasets/cpp.txt", "r")
for line in parsed_dataset:
    if line == None or len(line) <= 1:
        break
    line = line.split("\t\t\t\t\t\t")[1]
    tokens_all = re.split('(\W)', line)
    tokens = []
    for token in tokens_all:
        if token != '' and token != ' ' and token != '\n':
            tokens.append(token)
    for token in tokens:
        if token not in token_map:
            token_map[token] = counter
            counter += 1
parsed_dataset.close()

num_embeddings = len(token_map)
print("Num embeddings:", num_embeddings)
embedding_dim = 512
print("Embedding dim:", embedding_dim)
num_classes = 37
print("Num classes:", num_classes)

embeddings = torch.rand(num_embeddings, embedding_dim)

parsed_dataset = open("parsed_datasets/cpp.txt", "r")
X, Y = [], []
for line in parsed_dataset:
    if line == None or len(line) <= 1:
        break
    classes, parsed_code = line.split("\t\t\t\t\t\t")

    # Add to Y
    classes_list = classes.split(",")
    y = torch.zeros(num_classes)
    if classes != "":
        for c in classes_list:
            y[int(c)] = 1
    Y.append(y)

    # Add to X
    tokens_all = re.split('(\W)', parsed_code)
    tokens = []
    for token in tokens_all:
        if token != '' and token != ' ' and token != '\n':
            tokens.append(token)
    seq = []
    for token in tokens:
        idx = token_map[token]
        embedding = embeddings[idx]
        seq.append(embedding)
    seq = torch.stack(seq)
    X.append(seq)
parsed_dataset.close()

assert len(X) == len(Y)

device = torch.device('cuda:0')
for i in range(len(X)):
    X[i] = X[i].to(device=device)
    Y[i] = Y[i].to(device=device)

X_train, Y_train = X[:int(0.9*len(X))], Y[:int(0.9*len(Y))]
X_test, Y_test = X[int(0.9*len(X)):], Y[int(0.9*len(Y)):]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view((len(x), 1, -1))
        out = self.transformer_encoder(x)
        out = out.view((len(out), -1))
        out = self.linear(out[-1])
        out = self.sigmoid(out)
        return out

model = Model()
model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()
epochs = 10

for epoch in range(epochs):
    last_loss = None
    for x, y in zip(X_train, Y_train):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    print("Loss:", last_loss)

torch.save(model.state_dict(), "models/cpp_model.pt")