import os
from typing import List

import numpy as np
import pandas as pd


def traverse_data(handler):
    result = dict()
    for dataset in os.listdir('data'):
        for reducer in os.listdir(f'data/{dataset}'):
            if reducer == 'orig.csv':
                continue
            data_path = f'{dataset}/{reducer}'
            try:
                result[data_path] = handler(data_path)
                print(f'HANDLED {data_path}')
            except FileNotFoundError:
                print(f'SKIPPED {data_path}')
    return result


def read_orderings(data_path, num_accessors) -> List[List[int]]:
    orderings = list()
    for accessor_idx in range(num_accessors):
        accessor_path = f'data/{data_path}/accessor-{accessor_idx}.txt'
        with open(accessor_path, 'r') as fp:
            content = fp.readline()
            ordering = eval(content)
            orderings.append(ordering)
    return orderings


def read_meta_features(data_path) -> List[float]:
    features_path = f'data/{data_path}/features.txt'
    with open(features_path, 'r') as fp:
        content = fp.readline()
        return eval(content)


def read_partitions(data_path) -> List[List[int]]:
    partitions_path = f'data/{data_path}/partitions.csv'
    content = pd.read_csv(partitions_path, header=None)
    return content.values.tolist()


def read_gen_data(data_path) -> np.ndarray:
    gen_data_path = f'data/{data_path}/gen.csv'
    return pd.read_csv(gen_data_path, header=None).values
