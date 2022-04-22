import click
import csv
import math
import os
import random

import codeforces


@click.group()
def data():
    pass


def get_raw_data_file(dataset):
    path = '../data/'
    if dataset == 'codeforces_tags_java':
        return '{}{}.csv'.format(path, dataset)
    raise NameError('Invalid dataset')


@data.command()
@click.option('--data-dir', default='../data')
@click.option('--language', help='which language to scrape')
def codeforces_scrape(data_dir, language):
    codeforces.get_data(language)


@data.command()
@click.option('--data-dir', default='../data')
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

        for row in reader:
            problem_id = '{}/{}'.format(row[2], row[3])
            if problem_id in train_problem_ids:
                train_file.writerow([row[0], row[1]])
            elif problem_id in validation_problem_ids:
                validation_file.writerow([row[0], row[1]])
            elif problem_id in test_problem_ids:
                test_file.writerow([row[0], row[1]])

        train_file.close()
        validation_file.close()
        test_file.close()


if __name__ == '__main__':
    data()

