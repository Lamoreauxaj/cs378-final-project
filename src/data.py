import click
import csv
import math
import os
import random
from tree_sitter import Language, Parser

import codeforces
import code2seq


Language.build_library(
    'src/build/my-languages.so',
    [
        'src/vendor/tree-sitter-java',
        'src/vendor/tree-sitter-kotlin'
    ]
)

JAVA_LANGUAGE = Language('src/build/my-languages.so', 'java')
KOTLIN_LANGUAGE = Language('src/build/my-languages.so', 'kotlin')

def get_language(dataset):
    if dataset == 'codeforces_tags_java':
        return JAVA_LANGUAGE
    elif dataset == 'codeforces_tags_kotlin':
        return KOTLIN_LANGUAGE
    raise NameError('failed')


total_classes = 0
classes = []
classes_map = {}


def read_classes(data_dir, datasets):
    global classes, classes_map, total_classes
    classes_seen = set()
    for dataset in datasets:
        data_file = open(os.path.join(data_dir, '{}.csv'.format(dataset)), 'r')
        reader = csv.reader(data_file)
        for row in reader:
            tags = row[1].split(',')
            for tag in tags:
                if tag not in classes_seen and tag:
                    classes_seen.add(tag)
                    classes.append(tag)
                    classes_map[tag] = total_classes
                    total_classes += 1
        data_file.close()


class Example:
    def __init__(self, code, tags, parser):
        self.code = code
        self.lines = code.split('\n')
        self.tree = parser.parse(bytes(code, 'utf8'))
        self.tags = list(filter(lambda x: x, tags.split(',')))
        self.truth_ = None
        self.truth_ids_ = None


    def truth_ids(self):
       if self.truth_ids_ is None:
           self.truth_ids_ = []
           for tag in self.tags:
               if tag in classes_map:
                   self.truth_ids_.append(classes_map[tag])
       return self.truth_ids_


    def truth(self):
        if self.truth_ is None:
            self.truth_ = [0.0 for i in range(total_classes)]
            for tag in self.tags:
                if tag in classes_map:
                    self.truth_[classes_map[tag]] = 1.0
        return [self.truth_]


def get_train_data(dataset, data_dir, max_examples=None, validation=False):
    if validation:
        train_file = open(os.path.join(data_dir, '{}_validation.csv'.format(dataset)), 'r')
    else:
        train_file = open(os.path.join(data_dir, '{}_train.csv'.format(dataset)), 'r')
    reader = csv.reader(train_file)
    parser = Parser()
    parser.set_language(get_language(dataset))
    train_examples = []
    for row in reader:
        code = row[0]
        code = code.replace('\\n', '\n')
        example = Example(code, row[1], parser)
        if len(example.tags) > 0:
            train_examples.append(example)
        if max_examples is not None and len(train_examples) >= max_examples:
            break
    train_file.close()
    return train_examples


@click.group()
def data():
    pass


@data.command()
@click.option('--data-dir', default='data')
@click.option('--datasets', help='comma separated lists of which dataset to use')
@click.option('--max-train-examples', type=int, help='how many train examples to use')
@click.option('--save-dir')
@click.option('--cuda', default=False, type=bool)
def train_code2seq(data_dir, datasets, max_train_examples, save_dir, cuda):
    datasets = datasets.split(',')
    read_classes(data_dir, datasets)
    train_data = []
    for dataset in datasets:
        train_data += get_train_data(dataset, data_dir, max_examples=max_train_examples)
    validation_data = []
    for dataset in datasets:
        validation_data += get_train_data(dataset, data_dir, validation=True, max_examples=max_train_examples)
    model = code2seq.Code2Seq(num_classes=total_classes, save_dir=save_dir, cuda=cuda)
    model.train(train_data, validation_data)


def get_raw_data_file(dataset):
    path = 'data/'
    if dataset == 'codeforces_tags_java':
        return '{}{}.csv'.format(path, dataset)
    elif dataset == 'codeforces_tags_kotlin':
        return '{}{}.csv'.format(path, dataset)
    raise NameError('Invalid dataset')


@data.command()
@click.option('--data-dir', default='data')
@click.option('--language', help='which language to scrape')
def codeforces_scrape(data_dir, language):
    codeforces.get_data(language)


@data.command()
@click.option('--data-dir', default='data')
@click.option('--dataset', help='which dataset to use')
@click.option('--train-ratio', default=0.9, help='how much is training data')
@click.option('--validation-ratio', default=0.05, help='how much is validation data')
def preprocess(data_dir, dataset, train_ratio, validation_ratio):
    data_file = get_raw_data_file(dataset)

    with open(data_file, 'r') as f:
        reader = csv.reader(f)

        problems_set = set()

        for row in reader:
            problems_set.add('{}/{}'.format(row[2], row[3]))

    with open(data_file, 'r') as f:
        reader = csv.reader(f)

        print(len(problems_set))

        total_problems = len(problems_set)
        problem_ids = list(problems_set)
        random.shuffle(problem_ids)
        train_problems = math.floor(total_problems * train_ratio)
        validation_problems = math.floor(total_problems * validation_ratio)

        train_problem_ids = set(problem_ids[:train_problems])
        validation_problem_ids = set(problem_ids[train_problems:validation_problems + train_problems])
        test_problem_ids = set(problem_ids[validation_problems + train_problems:])

        train_file = open(os.path.join(data_dir, '{}_train.csv'.format(dataset)), 'w')
        validation_file = open(os.path.join(data_dir, '{}_validation.csv'.format(dataset)), 'w')
        test_file = open(os.path.join(data_dir, '{}_test.csv'.format(dataset)), 'w')
        train_file_writer = csv.writer(train_file)
        validation_file_writer = csv.writer(validation_file)
        test_file_writer = csv.writer(test_file)


        for row in reader:
            problem_id = '{}/{}'.format(row[2], row[3])
            if problem_id in train_problem_ids:
                train_file_writer.writerow([row[0], row[1]])
            elif problem_id in validation_problem_ids:
                validation_file_writer.writerow([row[0], row[1]])
            elif problem_id in test_problem_ids:
                test_file_writer.writerow([row[0], row[1]])

        train_file.close()
        validation_file.close()
        test_file.close()


if __name__ == '__main__':
    data()

