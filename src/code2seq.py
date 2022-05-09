import math
import random

import torch
import numpy as np
from torch import nn, optim


def traverse(tree, func):
    def recurse(cursor):
        func(cursor.node)
        if len(cursor.node.children) == 0:
            return
        l = len(cursor.node.children)
        cursor.goto_first_child()
        for idx in range(l):
            if idx > 0:
                cursor.goto_next_sibling()
            recurse(cursor)
        cursor.goto_parent()
    recurse(tree.walk())


def code_substring(example, start, end):
    res = ''
    for line in range(start[0], end[0] + 1):
        if line > start[0]:
            res += '\n'
        if line == start[0] and line == end[0]:
            res += example.lines[line][start[1]:end[1]]
        elif line == start[0]:
            res += example.lines[line][start[1]:]
        elif line == end[0]:
            res += example.lines[line][:end[1]]
        else:
            res += example.lines[line]
    return res


class Code2Seq(nn.Module):
    def __init__(self, train_examples, epochs=200, learning_rate=.001, num_classes=10):
        super(Code2Seq, self).__init__()
        self.ast_vocab = {'PAD': 0, 'UNK': 1}
        self.token_vocab = {'PAD': 0, 'UNK': 1}

        self.get_ast_vocabulary(train_examples)
        self.get_token_vocabulary(train_examples)

        self.num_paths = 200
        self.embedding_dim = 128
        self.hidden_size = 128
        self.proj_size = 128
        self.num_classes = num_classes

        self.ast_embedding = nn.Embedding(len(self.ast_vocab), self.embedding_dim)
        self.token_embedding = nn.Embedding(len(self.token_vocab), self.embedding_dim)

        self.path_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=2, dropout=0.2, bidirectional=True)
        self.proj = nn.Linear(4 * self.hidden_size + 2 * self.embedding_dim, self.proj_size)
        self.dropout = nn.Dropout(0.2)
        self.class_proj = nn.Linear(self.proj_size, self.num_classes)
        # self.softmax = nn.Softmax(dim=0)

        self.train(train_examples, epochs, learning_rate)


    def forward(self, paths, left_tokens, right_tokens):
        # print('input', paths.size(), left_tokens.size(), right_tokens.size())
        embedded_paths = self.ast_embedding(paths)
        embedded_left_tokens = self.token_embedding(left_tokens)
        embedded_right_tokens = self.token_embedding(right_tokens)
        # print('embedded', embedded_paths.size(), embedded_left_tokens.size(), embedded_right_tokens.size())
        encoded_paths, _ = self.path_lstm(embedded_paths)
        encoded_left_tokens = embedded_left_tokens.sum(dim=-2)
        encoded_right_tokens = embedded_right_tokens.sum(dim=-2)
        encoded_paths = torch.index_select(encoded_paths, dim=1, index=torch.tensor([0, encoded_paths.size(1) - 1]))
        encoded_paths = torch.flatten(encoded_paths, start_dim=1)
        # print('paths', encoded_paths.size())
        # print(encoded_left_tokens.size())
        # print(encoded_right_tokens.size())
        cat_data = torch.cat((encoded_paths, encoded_left_tokens, encoded_right_tokens), dim=1)
        # print('cat_data', cat_data.size())
        z = self.proj(cat_data)
        z = self.dropout(z)
        z = torch.tanh(z)
        h = torch.sum(z, dim=0) / torch.tensor([paths.size(0)]).unsqueeze(0)
        h = h.squeeze(0)
        # print('z', z.size())
        # print('h', h.size())
        output = self.class_proj(h)
        # output = torch.sigmoid(output)
        # print('output', output.size())
        # print(output)
        return output


    def split_token(self, token):
        tokens = token.split(' ')
        res = []
        for token in tokens:
            if '_' in token:
                res += token.lower().split('_')
            elif not token.isupper():
                l = 0
                for i in range(len(token) + 1):
                    l += 1
                    if i == len(token) or token[i].isupper():
                        res.append(token[i - l + 1:i + 1])
                        l = 0
            else:
                res.append(token)
        return res


    def get_ast_vocab_index(self, word):
        if word not in self.ast_vocab:
            return self.ast_vocab['UNK']
        return self.ast_vocab[word]


    def get_token_vocab_index(self, word):
        if word not in self.token_vocab:
            return self.token_vocab['UNK']
        return self.token_vocab[word]


    def get_ast_vocabulary(self, train_examples):
        for example in train_examples:
            def process(node):
                if node.type not in self.ast_vocab:
                    self.ast_vocab[node.type] = len(self.ast_vocab)
            traverse(example.tree, process)


    def get_token_vocabulary(self, train_examples):
        for example in train_examples:
            def process(node):
                if len(node.children) == 0:
                    token = code_substring(example, node.start_point, node.end_point)
                    tokens = self.split_token(token)
                    for subtoken in tokens:
                        if subtoken not in self.token_vocab:
                            self.token_vocab[subtoken] = len(self.token_vocab)
            traverse(example.tree, process)


    def select_paths(self, tree):
        leaf_nodes = []
        def process(node):
            if len(node.children) == 0:
                leaf_nodes.append(node)
        traverse(tree, process)
        chosen = set()
        paths = []
        for path_idx in range(self.num_paths):
            if path_idx >= len(leaf_nodes) * len(leaf_nodes) - 1 // 2:
                break
            path_start = None
            path_end = None
            while True:
                start = random.randint(0, len(leaf_nodes) - 1)
                end = random.randint(0, len(leaf_nodes) - 1)
                if start != end:
                    path_start = min(start, end)
                    path_end = max(start, end)
                    break
            paths.append((leaf_nodes[path_start], leaf_nodes[path_end]))
        return paths


    def process_input(self, example):
        paths = self.select_paths(example.tree)
        inp_paths = []
        inp_left_tokens = []
        inp_right_tokens = []
        max_path_len = 0
        max_left_token_len = 0
        max_right_token_len = 0

        for path in paths:
            seen = set()
            def h(node):
                return str(node.start_point) + str(node.end_point)

            left_nodes = []
            node = path[0]
            while node != example.tree.root_node:
                left_nodes.append(node)
                seen.add(h(node))
                node = node.parent
            left_nodes.append(node)
            seen.add(h(node))
            node = path[1]
            right_nodes = []
            while h(node) not in seen:
                right_nodes.append(node)
                node = node.parent
            nodes = []
            for node in left_nodes:
                nodes.append(node.type)
            for i in range(len(right_nodes) - 1, -1, -1):
                nodes.append(right_nodes[i].type)

            node_ids = [self.get_ast_vocab_index(word) for word in nodes]
            left_subtokens = self.split_token(code_substring(example, path[0].start_point, path[0].end_point))
            right_subtokens = self.split_token(code_substring(example, path[1].start_point, path[1].end_point))
            left_token_ids = [self.get_token_vocab_index(word) for word in left_subtokens]
            right_token_ids = [self.get_token_vocab_index(word) for word in left_subtokens]
            inp_paths.append(node_ids)
            inp_left_tokens.append(left_token_ids)
            inp_right_tokens.append(right_token_ids)
            max_path_len = max(max_path_len, len(node_ids))
            max_left_token_len = max(max_left_token_len, len(left_token_ids))
            max_right_token_len = max(max_right_token_len, len(right_token_ids))

        for row in inp_paths:
            while len(row) < max_path_len:
                row.append(0)
        for row in inp_left_tokens:
            while len(row) < max_left_token_len:
                row.append(0)
        for row in inp_right_tokens:
            while len(row) < max_right_token_len:
                row.append(0)

        inp_paths = torch.tensor(inp_paths)
        inp_left_tokens = torch.tensor(inp_left_tokens)
        inp_right_tokens = torch.tensor(inp_right_tokens)
        return inp_paths, inp_left_tokens, inp_right_tokens


    def train(self, train_examples, epochs, learning_rate):

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        examples_with_paths = [(example, self.process_input(example)) for example in train_examples]

        for epoch_idx in range(epochs):

            print('epoch', epoch_idx)

            random.shuffle(examples_with_paths)
            total_loss = 0

            correct = 0
            negative_acc = 0

            for (example, (paths, left_tokens, right_tokens)) in examples_with_paths:
                optimizer.zero_grad()
                paths, left_tokens, right_tokens = self.process_input(example)
                predicted = self.forward(paths, left_tokens, right_tokens).unsqueeze(0)
                target = torch.tensor(example.truth())

                prob_predicted = torch.log_softmax(predicted, dim=1)
                # print(prob_predicted)

                local_correct = 0
                total_needed = 0

                negatives = 0
                true_negatives = 0
                # print(target)
                for idx in range(target.size(1)):
                    # print(target[0][idx].item())
                    above = prob_predicted[0][idx].item() > math.log(0.5)
                    if target[0][idx].item() >= .9:
                        total_needed += 1
                        # print(prob_predicted[0][idx].item())
                        if above:
                            local_correct += 1
                    else:
                        true_negatives += 1
                        if not above:
                            negatives += 1
                    # if abs(target[0][idx] - prob_predicted[0][idx]) < 0.2:
                    #     local_correct += 1
                correct += local_correct / total_needed
                negative_acc += negatives / true_negatives



                # print('predicted', predicted, predicted.size())
                # print('target', target, target.size())
                loss = loss_func(predicted, target)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            correct /= len(train_examples)
            negative_acc /= len(train_examples)

            print(total_loss, correct, negative_acc)





